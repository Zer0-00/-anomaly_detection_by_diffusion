import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Union
import os
import csv
from functools import partial

ImageClass = Union[torch.Tensor,np.ndarray]

def dice_coeff(
    targets:ImageClass, 
    images:ImageClass, 
    epsilon=1e-6
):
    """
    calculate dice coefficient

    Args:
        images (torch.Tensor|np.ndarray): input image of (N,1,H,W) or (N,H,W,1)
        targets (torch.Tensor|np.ndarry): ground truth, should share the same shape as input
        epsilon (optional): a small number added to the denominator. Defaults to 1e-6.
    """
    assert images.shape == targets.shape and type(images) == type(targets),\
         "the input and target images should share the same shape and type"
    dice = 0
    dot_fn = torch.dot if type(images) == torch.Tensor else np.dot
    for image, target in zip(images, targets):
        dot = dot_fn(image.reshape(-1), target.reshape(-1))
        sum = image.sum() + target.sum()
        dice += (2 * dot + epsilon) / (sum + epsilon)
        
    dice = dice / images.shape[0]
    return dice

def region_specific_metrics(
    targets:ImageClass, 
    images:ImageClass, 
    func,
    region_type='WT',
    **func_kwargs
):
    assert region_type in ["ET", "TC", "WT"], "region type should be one of ET, TC, WT"
    if region_type == 'ET':
        masks = (targets == 1) * 1
    elif region_type == "TC":
        masks = (((targets == 1) + (targets == 4)) > 0) * 1
    else:
        masks = (((targets == 1) + (targets == 2) + (targets == 4))) > 0 * 1
        
    return func(masks, images, **func_kwargs)

def AUROC(
    targets:ImageClass, 
    images:ImageClass, 
):
    """
    calculate AUROC

    Args:
        images (torch.Tensor|np.ndarray): input image of (N,1,H,W) or (N,H,W,1)
        targets (torch.Tensor|np.ndarray): ground truth, should share the same shape as input
    """
    assert images.shape == targets.shape and type(images) == type(targets),\
         "the input and target images should share the same shape and type"
         
    
    if isinstance(images, torch.Tensor):     
        targets = targets.detach().cpu().to(torch.uint8).numpy().flatten().squeeze()
        images = images.detach().cpu().numpy().flatten().squeeze()
    else:
        targets = targets.flatten().squeeze()
        images = images.flatten().squeeze()
    try: 
        score = roc_auc_score(targets, images)
    except ValueError:
        score = -1
        
    return score

def min_max_scale(image:ImageClass):
    return (image - image.min())/(image.max()-image.min())

def nonzero_masking(images:ImageClass, targets:ImageClass):
    """
    masking targets according to the non-zero pixels of images
     
    p.s: non-zero is not mean the value of pixels is zero(sometimes tensor can be among [-1,1]),
         so pixels are reagarded as non-zero when the value of them is larger than images.min()
    
    Args:
        images (torch.Tensor|np.ndarray): input image of (N,1,H,W) or (N,H,W,1)
        targets (torch.Tensor|np.ndarray): ground truth, should share the same shape as input
    """
    assert type(images) == type(targets),\
        "the input and target images should share the same and type"
    if isinstance(images, torch.Tensor):
        feature_dim = 1
        assert images.shape[2,3] == targets.shape[2,3],\
            "the input and target images should share the same shape(H,W)"  
            
    else:
        feature_dim = 1
        assert images.shape[1,2] == targets.shape[1,2],\
            "the input and target images should share the same shape(H,W)"
            
    mask = ((images > images.min() * 1.0).sum(feature_dim) == 4) * 1.0
    
    targets = targets * mask + targets.min() * (1 - mask)
    
    return targets
        
    

class BratsEvaluator():
    def __init__(
        self,
        data_folder,
        metrics,
        mask_fn=None,
    ):
        self.data_folder = data_folder
        self.metrics = metrics
        
        self.mask_fn = mask_fn if mask_fn is not None else min_max_scale
        self.data_files = [file_name for file_name in os.listdir(self.data_folder) if file_name.endswith(".npy")]
        
    def evaluate(self,output_dir):
        with open(os.path.join(output_dir,"metrics.csv"), 'w', newline='') as csvfile:
            fieldnames = list(self.metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for file_name in self.data_files:
                file_dir = os.path.join(self.data_folder, file_name)
                data = np.load(file_dir)
                
                img = data[0,:,:,:4]*1.0
                seg = np.expand_dims(data[0,:,:,4], axis=(0,1))
                generated = data[0,:,:,5:]*1.0
                pred = np.expand_dims(np.sum((generated-img)**2, axis=2), axis=(0,1))
                pred = nonzero_masking(img, pred)
                pred = self.mask_fn(pred)
                
                metrics_img = {metric: metric_fn(seg,pred) for metric, metric_fn in self.metrics.items()}
                writer.writerow(metrics_img)
                
def evaluate_Brat(data_folder, output_dir):
    metrics = {
        "DICE_ET": partial(region_specific_metrics, func=dice_coeff, region_type="ET"),
        "DICE_TC": partial(region_specific_metrics, func=dice_coeff, region_type="TC"),
        "DICE_WT": partial(region_specific_metrics, func=dice_coeff, region_type="WT"),
        "AUROC_ET": partial(region_specific_metrics, func=AUROC, region_type="ET"),
        "AUROC_TC": partial(region_specific_metrics, func=AUROC, region_type="TC"),
        "AUROC_WT": partial(region_specific_metrics, func=AUROC, region_type="WT"),
    }
    
    evaluator = BratsEvaluator(
        data_folder=data_folder,
        metrics=metrics,
    )
    
    evaluator.evaluate(output_dir)
    
if __name__ == "__main__":
    evaluate_Brat('./output/anomaly_detection','./output/anomaly_detection')
    