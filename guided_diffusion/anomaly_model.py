from .respace import SpacedDiffusion
import torch
from .unet import EncoderUNetModel, UNetModel

def mse_map(image, target):
    return torch.sum((image - target)**2, dim=1)

class AnomalyDiffusion(SpacedDiffusion):
    def __init__(self, max_t, **kwargs):
        kwargs["use_timesteps"] = self.filter_timesteps(kwargs["use_timesteps"], max_t)
        self.max_origin_t = max(kwargs["use_timesteps"])
        
        super().__init__(**kwargs)
    
    def ddpm_anomaly_detection(
        self,
        model,
        img,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        detection_fn=None,
    ):
        """
        Generate anomaly detection result from model

        Args:
            model: the model module.
            img: input image of shape(N,C,H,W)
            clip_denoised (optional): if True, clip x_start predictions to [-1, 1]. Defaults to True.
            denoised_fn (optional): if not None, a function which applies to the
                                    x_start prediction before it is used to sample. Defaults to None.
            cond_fn (optional): if not None, this is a gradient function that acts
                                similarly to the model. Defaults to None.
            model_kwargs (optional): if not None, a dict of extra keyword arguments to
                                     pass to the model. This can be used for conditioning. Defaults to None.
            device (optional): if specified, the device to create the samples on.
                               If not specified, use a model parameter's device. Defaults to None.
            progress (optional): if True, show a tqdm progress bar. Defaults to False.
            detection_fn (optional): if specified, a function which applies to the sample yeilded by the model
                                     to generate the detection map.If not specified, default use mse.
                                     
            
        return a non-differentiable batch of detection map.
        """
        if detection_fn is None:
            detection_fn = mse_map
        
        max_t = torch.tensor(self.num_timesteps, device=device)
        print(max_t)
        img_noised = self.q_sample(x_start=img, t=max_t)
        
        for sample in self.p_sample_loop_progressive(
            model,
            img.shape,
            noise=img_noised,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            img_generated = sample["sample"]
        
        detection_map = detection_fn(img_generated, img)
        
        output = {
            "input": img,
            "generated_image": img_generated,
            "detection_map": detection_map
        }
        
        return output
    
    def ddim_anomaly_detection(
        self,
        model,
        img,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        detection_fn=None,
    ):
        """
        Generate anomaly detection result from model

        Args:
            model: the model module.
            img: input image of shape(N,C,H,W)
            clip_denoised (optional): if True, clip x_start predictions to [-1, 1]. Defaults to True.
            denoised_fn (optional): if not None, a function which applies to the
                                    x_start prediction before it is used to sample. Defaults to None.
            cond_fn (optional): if not None, this is a gradient function that acts
                                similarly to the model. Defaults to None.
            model_kwargs (optional): if not None, a dict of extra keyword arguments to
                                     pass to the model. This can be used for conditioning. Defaults to None.
            device (optional): if specified, the device to create the samples on.
                               If not specified, use a model parameter's device. Defaults to None.
            progress (optional): if True, show a tqdm progress bar. Defaults to False.
            detection_fn (optional): if specified, a function which applies to the sample yeilded by the model
                                     to generate the detection map.If not specified, default use mse.
                                     
            
        return a non-differentiable batch of detection map.
        """
        if detection_fn is None:
            detection_fn = mse_map
        
        max_t = torch.tensor(self.num_timesteps-1, device=device)
        img_noised = self.q_sample(x_start=img, t=max_t)
        
        for sample in self.ddim_sample_loop_progressive(
            model,
            img.shape,
            noise=img_noised,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            img_generated = sample["sample"]
        
        detection_map = detection_fn(img_generated, img)
        
        output = {
            "input": img,
            "generated_image": img_generated,
            "detection_map": detection_map
        }
        
        return output
    
    def filter_timesteps(self, origin_steps, max_t):
        
        if max_t > 0:
            filtered_steps = {step for step in origin_steps if step <= max_t}
        else:
            #if max_t <= 0, then we don't need to filter
            filtered_steps = set(origin_steps)
        return filtered_steps    
        
    def visualize_images(self, model, img, groundtruth=None):
        """
        Visualize results of anomaly detections
        """
        pass
    
class DecoupledDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
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
        class_cond=False,
        emb_combination='plus',
        extra_emb_dim=None,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        #save model parameters
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.class_cond = class_cond
        
        extra_emb_dim = extra_emb_dim if extra_emb_dim is not None else model_channels * 4
                
        self.encoder = EncoderUNetModel(
            image_size=self.image_size,
            in_channels=self.in_channels,
            model_channels=encoder_model_channels,
            out_channels=extra_emb_dim,
            num_res_blocks=encoder_num_res_blocks,
            attention_resolutions=encoder_attention_resolutions,
            channel_mult=encoder_channel_mult,
            use_fp16=use_fp16,
            num_head_channels=encoder_num_head_channels,
            use_scale_shift_norm=encoder_use_scale_shift_norm,
            resblock_updown=encoder_resblock_updown,
            pool=pool,
            num_classes=2 if self.class_cond else None,
            input_time=False,
        )
        
        self.denoised = UNetModel(
            image_size=self.image_size,
            in_channels=self.in_channels,
            model_channels=model_channels,
            out_channels=self.out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=self.dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            emb_combination=emb_combination,
            extra_emb_dim=extra_emb_dim,
        )
        
    def forward(self, x, timesteps, x0, y=None):
        extra_emb = self.get_embbed(x0, y=y)
        denoised_img = self.denoised(x, timesteps, extra_emb=extra_emb)
        
        return denoised_img
    
    def get_embbed(self, x, y=None):
        return self.encoder(x, None, y=y)
    
    def predict_with_Z(self, x, timesteps, z):
        denoised_img = self.denoised(x, timesteps, extra_emb=z)
        
        return denoised_img