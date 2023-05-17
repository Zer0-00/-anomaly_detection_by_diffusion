"""
Train a Linear Classifier for image curing and save median confidence for normal image.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import load_data
from guided_diffusion.fp16_util import MixedPrecisionTrainer
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
    
    z_state = np.load(args.z_state_path)
    z_mean = th.Tensor(z_state['z_mean']).to(device=dist_util.dev())
    z_std = th.Tensor(z_state['z_std']).to(device=dist_util.dev())
    
    model.to(dist_util.dev())
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = th.nn.Linear(model.encoder.out_channels, 1)
    classifier.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_labels=True,
        dataset=args.dataset,
        deterministic=False,
    )
    
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            class_labels=True,
            dataset=args.dataset,
            deterministic=True,
        )
    
    dist_util.sync_params(model.parameters())
    dist_util.sync_params(classifier.parameters())
    
    mp_trainer = MixedPrecisionTrainer(
        model=classifier, use_fp16=args.use_fp16, initial_lg_loss_scale=16.0
    )
    logger.log("creating optimiser")
    opt = Adam(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    
    classifier = DDP(
        classifier,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    
    logger.log("classifier training")
    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        
        for i, (sub_batch, sub_labels) in enumerate(
            split_microbatches(args.microbatch, batch, labels)
        ):  
            model_kwargs = {}
            if args.class_cond:
                model_kwargs["y"] = sub_labels
            z = model.get_embbed(sub_batch, **model_kwargs)
            z = (z - z_mean)/z_std
            preds = classifier(z)
            
            loss = F.binary_cross_entropy_with_logits(preds.squeeze(), sub_labels.to(th.float32), reduction="none")
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_accuracy(
                preds, sub_labels, reduction="none"
            )
            log_loss_dict(losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
    
    for step in range(args.iterations):
        logger.logkv("step", step)
        logger.logkv(
            "samples",
            (step  + 1) * args.batch_size * dist.get_world_size(),
        )
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with classifier.no_sync():
                    classifier.eval()
                    forward_backward_log(val_data, prefix="val")
                    classifier.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step)

    #calculate median prediction
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_labels=True,
        dataset=args.dataset,
        deterministic=False,
        limited_num=-1
    )
    lgs_healthy = find_lgs(model, classifier, data, z_mean, z_std, 0)
    lgs_healthy = np.concatenate(lgs_healthy, axis=0).mean(axis=0)
    
    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step)
        save_dir = os.path.join(logger.get_dir(), 'median_logist')
        np.savez(save_dir, lgs_healthy=lgs_healthy)

    dist.barrier()
    logger.log("Linear Classifier training finished")
    
def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))
    
def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def compute_accuracy(preds, labels, reduction="mean"):
    pred_labels = preds > 0
    if reduction == "mean":
        return (pred_labels == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (pred_labels == labels[:, None]).float().sum(dim=-1)
    
def log_loss_dict(losses):
    for k, v in losses.items():
        logger.logkv_mean(k, v.mean().item())

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)
            
def find_lgs(model, classifier, data, z_mean, z_std, class_label=0):
    lgs_healthy = []
    with th.no_grad():
        for batch, extra in data:
            labels = extra["y"].to(dist_util.dev())
            batch = batch.to(dist_util.dev())
            
            model_kwargs = {}
            if model.class_cond:
                model_kwargs['y'] = labels
            z = model.get_embbed(batch, **model_kwargs)
            z = (z - z_mean)/z_std
            preds = classifier(z)
            
            lg_healthy = preds[th.where(labels == class_label)]
            gathered_lg = [th.zeros_like(lg_healthy) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_lg, lg_healthy)  # gather not supported with NCCL
            lgs_healthy.extend([lg_healthy.cpu().numpy() for lg_healthy in gathered_lg])
    
    return lgs_healthy

def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        batch_size=4,
        microbatch=-1,
        dataset="brats2020",
        output_dir="./output/configs3/Z/",
        model_path="",
        iterations=10000,
        log_interval=10,
        eval_interval=5,
        save_interval=100,
        lr=3e-4,
        weight_decay=0.0,
        z_state_path=""
    )
    defaults.update(decoupled_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--cfg", default="zGenerate_3", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
