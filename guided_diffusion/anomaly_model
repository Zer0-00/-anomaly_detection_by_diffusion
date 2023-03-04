from .respace import SpacedDiffusion
import torch

class anomaly_diffusion_model(SpacedDiffusion):
    def __init__(self, max_t, **kwargs):
        self.max_t = max_t
        kwargs["use_timesteps"] = self.filter_timesteps(kwargs["use_timesteps"], self.max_t)
        
        super().__init__(**kwargs)
    
    
    def generate_image_translator(
        self, 
        model, 
        img,
        t, 
        use_ddim=False):
        """
        Gnerate a image translator that translate anomalous images to normal ones.
        """
        pass
    
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
            detection_fn = self.mse_map
        
        img_noised = self.q_sample(x_start=img, t=self.max_t)
        
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
            detection_fn = self.mse_map
        
        img_noised = self.q_sample(x_start=img, t=self.max_t)
        
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
        
    def mse_map(self, image, target):
        return torch.sum((image - target)**2, dim=1)
    
    def filter_timesteps(origin_steps, max_t):
        
        filtered_steps = {step for step in origin_steps if step <= max_t}
        return filtered_steps    
        
    def visualize_images(self, model, img, groundtruth=None):
        """
        Visualize results of anomaly detections
        """
        pass