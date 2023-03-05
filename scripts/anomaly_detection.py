"""
anomaly detection with guided diffusion
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import time

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.anomaly_utils import create_anomaly_model_and_diffusion
from guided_diffusion.dataset import load_data

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_anomaly_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            dataset=args.dataset,
            deterministic=True
        )
    
    
    
    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("testing...")
    
    all_images = []
    sample_fn = (
        diffusion.ddpm_anomaly_detection if not args.use_ddim else diffusion.ddim_anomaly_detection
    )
    
    start_time = time.time()
    for imgs, extra in data:
        model_kwargs = {}
        classes = th.zeros(
            size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        img_batch = imgs.to(dist_util.dev())
        
        sample = sample_fn(
            model_fn,
            img_batch,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )["generated_image"]
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    
    arr = np.concatenate(all_images, axis=0)
    time_taken = time.time() - start_time
    logger.log(f"Take {time_taken}s to test {len(arr)} samples")
    
    if dist.get_rank() == 0:
        logger.log(f"saving to {logger.get_dir()}")
        for idx, img, extra in enumerate(data):
            
            seg = float2uint(extra["seg"]).squeeze().numpy()
            img = float2uint(img).squeeze.numpy()
            generated = arr[idx]
            
            save_arr = np.concatenate([img,seg,generated],axis=0)
            
            out_path = os.path.join(logger.get_dir(), f"samples_{idx}.npy")
            np.savez(out_path, save_arr)

    dist.barrier()
    logger.log("anomaly detection complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        output_dir="./output/anomaly_detection"
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def float2uint(input:th.Tensor):
    input = (input * 255).clamp(0, 255).to(th.uint8)
    input = input.permute(0, 2, 3, 1)
    input = input.contiguous()
    return input

if __name__ == "__main__":
    main()