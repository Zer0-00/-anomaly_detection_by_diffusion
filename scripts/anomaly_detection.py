"""
anomaly detection with guided diffusion
"""

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import tqdm

from guided_diffusion import dist_util, logger, utils
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
    args.__dict__.update(utils.load_parameters(args))
    
    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_anomaly_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        max_t=args.max_t,
    )
    print(args.model_path)
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
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            dataset=args.dataset,
            deterministic=True,
            limited_num=-1,
            test=True,
        )
    
    
    
    def cond_fn(x, t, y=None,mask=None):
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
    
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    start.record()
    for imgs, extra in tqdm.tqdm(data):
        model_kwargs = {}
        # classes = th.randint(
        #     low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
        # )
        classes = th.zeros(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int64)
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
    end.record()
    th.cuda.synchronize()
    th.cuda.current_stream().synchronize()
    time_taken = start.elapsed_time(end)
    logger.log(f"Take {time_taken}s to test {len(arr)} samples")
    
    if dist.get_rank() == 0:
        logger.log(f"saving to {logger.get_dir()}")
        for idx, (img, extra) in enumerate(data):
            
            seg = float2uint(extra["seg"]).numpy()
            img = float2uint(img).numpy()
            generated = arr[idx][None,...]
            
            save_arr = np.concatenate([img,seg,generated],axis=3)
            
            out_path = os.path.join(logger.get_dir(), f"samples_{idx}.npy")
            np.save(out_path, save_arr)

    dist.barrier()
    logger.log("anomaly detection complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        max_t=500,
        output_dir="./output/anomaly_detection",
        dataset="Brats2020",
        data_dir = "/home/xuehong/Datasets/Brats_Processed_Split/val"
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--cfg", default="classifier_1", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser

def float2uint(input:th.Tensor):
    input = (input * 255).clamp(0, 255).to(th.uint8)
    input = input.permute(0, 2, 3, 1)
    input = input.contiguous()
    return input

if __name__ == "__main__":
    main()
