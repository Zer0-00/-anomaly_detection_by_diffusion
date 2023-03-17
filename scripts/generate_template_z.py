"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch as th
import torch.distributed as dist
import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import load_data
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,

)
from guided_diffusion.utils import load_parameters
from guided_diffusion.anomaly_utils import create_semantic_encoder, semantic_encoder_defaults


def main():
    args = create_argparser().parse_args()
    args.__dict__.update(load_parameters(args))
    
    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)

    logger.log("loading model")
    model = create_semantic_encoder(
        **args_to_dict(args, semantic_encoder_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.encoder_path, map_location='cpu')
    )
    
    model.to(dist_util.dev())
    model.eval()

    if args.encoder_use_fp16:
        model.convert_to_fp16()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_labels=True,
        dataset=args.dataset
    )

    all_zs = []
    all_labels = []
    
    logger.log("Z generating...")
    with th.no_grad():
        for imgs, extra in tqdm.tqdm(data):

            labels = extra["y"]
            img_batch = imgs.to(dist_util.dev())
            
            z = model((img_batch, th.zeros_like((img_batch.shape[0],1), device=dist_util.dev())))
            
            gathered_zs = [th.zeros_like(z) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_zs, z)  # gather not supported with NCCL
            all_zs.extend([z.cpu().numpy() for z in gathered_zs])
            
            gathered_labels = [th.zeros_like(z) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, labels)  # gather not supported with NCCL
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        all_zs = np.concatenate(all_zs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    
    if dist.get_rank() == 0:
        logger.log(f"saving to {logger.get_dir()}")
        out_path = os.path.join(logger.get_dir(), "zs_and_labels.npy")
        np.savez(out_path, all_zs=all_zs, all_labels=all_labels)
        

    dist.barrier()
    logger.log("Z generating complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        dataset="brats2020",
        output_dir="./output/classifier",
        encoder_path=""
    )
    defaults.update(semantic_encoder_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--cfg", default="encoder_4", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
