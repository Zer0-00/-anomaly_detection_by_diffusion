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
from guided_diffusion.anomaly_utils import create_decoupled_model, decoupled_diffusion_defaults


def main():
    args = create_argparser().parse_args()
    args.__dict__.update(load_parameters(args))
    
    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)

    logger.log("loading model")
    model = create_decoupled_model(
        **args_to_dict(args, decoupled_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )
    
    model.to(dist_util.dev())
    model.eval()

    if args.use_fp16:
        model.convert_to_fp16()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_labels=True,
        dataset=args.dataset,
        deterministic=True,
        limited_num=-1
    )

    all_zs = []
    all_labels = []
    
    logger.log("Z generating...")
    with th.no_grad():
        for imgs, extra in tqdm.tqdm(data):

            labels = extra["y"].to(dist_util.dev())
            img_batch = imgs.to(dist_util.dev())
            
            z = model.get_embbed((img_batch))
            
            gathered_zs = [th.zeros_like(z) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_zs, z)  # gather not supported with NCCL
            all_zs.extend([z.cpu().numpy() for z in gathered_zs])
            
            gathered_labels = [th.zeros_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, labels)  # gather not supported with NCCL
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        all_zs = np.concatenate(all_zs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    
    if dist.get_rank() == 0:
        normal_mask = np.where(all_labels == 0)
        abnormal_mask = np.where(all_labels == 1)
        normal_meanZ = all_zs[normal_mask].mean(axis=0)
        abnormal_meanZ = all_zs[abnormal_mask].mean(axis=0)
    
        save_dir = os.path.join(logger.get_dir(), 'templates')
        np.savez(save_dir, normalZ=normal_meanZ, abnormalZ=abnormal_meanZ)
        
        
        logger.log(f"saving to {logger.get_dir()}")
        if args.save_allz:
            out_path = os.path.join(logger.get_dir(), "zs_and_labels.npy")
            np.savez(out_path, all_zs=all_zs, all_labels=all_labels)
        

    dist.barrier()
    logger.log("Z generating complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        batch_size=4,
        microbatch=-1,
        dataset="brats2020",
        output_dir="./output/configs3/Z/",
        model_path="",
        save_allz=False,
    )
    defaults.update(decoupled_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--cfg", default="zGenerate_3", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
