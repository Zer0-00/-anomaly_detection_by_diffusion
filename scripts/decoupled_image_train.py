"""
Train a diffusion model on images.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    NUM_CLASSES
)
from guided_diffusion.anomaly_utils import (
    anomaly_diffusion_and_model_defaults,
    semantic_encoder_defaults,
    create_anomaly_model_and_diffusion,
    create_semantic_encoder,
)
from guided_diffusion.train_util import DecoupledDiffusionTrainLoop
from guided_diffusion.utils import load_parameters


def main():
    args = create_argparser().parse_args()
    args.__dict__.update(load_parameters(args))

    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_anomaly_model_and_diffusion(
        **args_to_dict(args, anomaly_diffusion_and_model_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    semantic_encoder = create_semantic_encoder(
        **args_to_dict(args, semantic_encoder_defaults().keys)
    )
    
    #load weights from a pretrained classifier except the last layer
    state_dict =  dist_util.load_state_dict(args.encoder_path, map_location='cpu')
    for k in state_dict:
        if k.startswith('pool'):
            state_dict.pop(k)
    
    semantic_encoder.load_state_dict(state_dict,strict=False)
    for name, param in semantic_encoder.named_parameters():
        if 'pool' not in name:
            param.requires_grad = False

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        class_labels=False
    )

    logger.log("training...")
    DecoupledDiffusionTrainLoop(
        model=model,
        diffusion=diffusion,
        semantic_encoder=semantic_encoder,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        iterations=args.iterations,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser(configs=None):
    defaults = dict(
        data_dir="",
        encoder_path="./model/classifier.pt",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        iterations=500000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="brats2020",
        output_dir="./output/diffusion"
    )
    defaults.update(anomaly_diffusion_and_model_defaults)
    defaults.update(semantic_encoder_defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--cfg", default="image_1", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
