from .script_util import create_model,diffusion_defaults
from . import gaussian_diffusion as gd
from .respace import space_timesteps
from .anomaly_model import AnomalyDiffusion,DecoupledDiffusionModel
from .unet import EncoderUNetModel

def create_anomaly_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    max_t,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    in_channels=3
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        in_channels=in_channels,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    
    diffusion = create_anomaly_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        max_t=max_t,
    )
    
    return model, diffusion

def create_decoupled_model_and_diffusion(
    *,
    image_size,
    in_channels,
    model_channels,
    num_res_blocks,
    attention_resolutions,
    encoder_model_channels,
    encoder_num_res_blocks,
    encoder_attention_resolutions,
    dropout=0,
    channel_mult=(1, 2, 4, 8),
    use_checkpoint=False,
    use_fp16=False,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    resblock_updown=False,
    use_new_attention_order=False,
    encoder_channel_mult=(1, 2, 4, 8),
    encoder_num_head_channels=-1,
    encoder_use_scale_shift_norm=False,
    encoder_resblock_updown=False,
    pool='adaptive',
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    max_t=500,  
):
    diffusion = create_anomaly_gaussian_diffusion(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    sigma_small=sigma_small,
    noise_schedule=noise_schedule,
    use_kl=use_kl,
    predict_xstart=predict_xstart,
    rescale_timesteps=rescale_timesteps,
    rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
    max_t=max_t,
    )
    
    model = create_decoupled_model(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        encoder_model_channels=encoder_model_channels,
        encoder_num_res_blocks=encoder_num_res_blocks,
        encoder_attention_resolutions=encoder_attention_resolutions,
        dropout=dropout,
        learn_sigma=learn_sigma,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        encoder_channel_mult=encoder_channel_mult,
        encoder_num_head_channels=encoder_num_head_channels,
        encoder_use_scale_shift_norm=encoder_use_scale_shift_norm,
        encoder_resblock_updown=encoder_resblock_updown,
        pool=pool,
    )

    return model, diffusion

def create_anomaly_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    max_t=500,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return AnomalyDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        max_t=max_t
    )
    
def create_decoupled_model(
    image_size,
    in_channels,
    model_channels,
    num_res_blocks,
    attention_resolutions,
    encoder_model_channels,
    encoder_num_res_blocks,
    encoder_attention_resolutions,
    dropout=0,
    learn_sigma=False,
    channel_mult=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    resblock_updown=False,
    use_new_attention_order=False,
    encoder_channel_mult=None,
    encoder_num_head_channels=-1,
    encoder_use_scale_shift_norm=False,
    encoder_resblock_updown=False,
    pool='adaptive',
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    encoder_channel_mult = encoder_channel_mult
    
    if encoder_channel_mult == "":
        if image_size == 512:
            encoder_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            encoder_channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            encoder_channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            encoder_channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        encoder_channel_mult = tuple([int(num) for num in encoder_channel_mult.split(",")])

    encoder_attention_ds = []
    for res in encoder_attention_resolutions.split(","):
        encoder_attention_ds.append(image_size // int(res))
        
    return DecoupledDiffusionModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=(in_channels if not learn_sigma else in_channels*2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        encoder_model_channels=encoder_model_channels,
        encoder_num_res_blocks=encoder_num_res_blocks,
        encoder_attention_resolutions=tuple(encoder_attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        encoder_channel_mult=encoder_channel_mult,
        encoder_num_head_channels=encoder_num_head_channels,
        encoder_use_scale_shift_norm=encoder_use_scale_shift_norm,
        encoder_resblock_updown=encoder_resblock_updown,
        pool=pool,
    )
    
def create_semantic_encoder(
    image_size,
    emb_dim,
    encoder_use_fp16,
    encoder_width,
    encoder_depth,
    encoder_attention_resolutions,
    encoder_use_scale_shift_norm,
    encoder_resblock_updown,
    encoder_pool,
    in_channels=3,
    encoder_channel_mult=None,
    num_head=64,
):
    channel_mult = encoder_channel_mult
    
    if channel_mult is None:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple([int(num) for num in channel_mult.split(",")])

    attention_ds = []
    for res in encoder_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=encoder_width,
        out_channels=emb_dim,
        num_res_blocks=encoder_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=encoder_use_fp16,
        num_head_channels=num_head,
        use_scale_shift_norm=encoder_use_scale_shift_norm,
        resblock_updown=encoder_resblock_updown,
        pool=encoder_pool,
    )
    
    
def decoupled_diffusion_defaults():
    """
    defaults of decoupled diffusion for Brats.
    """
    
    defaults = dict(
    image_size=256,
    in_channels=4,
    model_channels=128,
    learn_sigma=False,
    num_res_blocks=2,
    attention_resolutions="16",
    encoder_model_channels=32,
    encoder_num_res_blocks=4,
    encoder_attention_resolutions="32,16,8",
    dropout=0.0,
    channel_mult="",
    use_checkpoint=False,
    use_fp16=False,
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_new_attention_order=False,
    encoder_channel_mult="",
    encoder_num_head_channels=64,
    encoder_use_scale_shift_norm=True,
    encoder_resblock_updown=True,
    pool='adaptive',
    )
    
    return defaults

def anomaly_diffusion_defaults():
    defaults = dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        max_t=500
    )
    return defaults

def model_defaults():
    defaults = dict(
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    return defaults

def semantic_encoder_defaults():
    defaults = dict(
        image_size=64,
        encoder_use_fp16=False,
        encoder_width=128,
        encoder_depth=2,
        encoder_attention_resolutions="32,16,8",  # 16
        encoder_use_scale_shift_norm=True,  # False
        encoder_resblock_updown=True,  # False
        encoder_pool="attention",
        in_channels=3,
        encoder_channel_mult="1,1,2,2,4,4",
        num_head=64
    )
    return defaults

def decoupled_diffusion_and_diffusion_defaults():
    defaults = decoupled_diffusion_defaults()
    defaults.update(anomaly_diffusion_defaults())
    return defaults
    
def anomaly_diffusion_and_model_defaults():
    defaults = anomaly_diffusion_defaults
    defaults.update(model_defaults())
    return defaults