import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Union
import os
import csv
from functools import partial

def dice_coeff(
    targets:Union[torch.Tensor,np.ndarray], 
    images:Union[torch.Tensor,np.ndarray], 
    epsilon=1e-6
):
    """
    calculate dice coefficient

    Args:
        images (torch.Tensor|np.ndarray): input image of (N,1,H,W)
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

def dice_specific(
    targets:Union[torch.Tensor,np.ndarray], 
    images:Union[torch.Tensor,np.ndarray], 
    epsilon=1e-6,
    region_type='WT'
):
    assert region_type in ["ET", "TC", "WT"], "region type should be one of ET, TC, WT"
    if region_type == 'ET':
        masks = (targets == 1) * 1
    elif region_type == "TC":
        masks = ((targets == 1) + (targets == 4)) > 0 * 1
    else:
        masks = ((targets == 1) + (targets == 2) + (targets == 4)) > 0 * 1
        
    return dice_coeff(masks, images, epsilon=epsilon)

def AUROC(
    targets:Union[torch.Tensor,np.ndarray], 
    images:Union[torch.Tensor,np.ndarray], 
):
    """
    calculate AUROC

    Args:
        images (torch.Tensor|np.ndarray): input image of (N,1,H,W)
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

class Brats_Evaluator():
    def __init__(
        self,
        data_folder,
        metrics,
    ):
        self.data_folder = data_folder
        self.metrics = metrics
        
        self.data_files = [file_name for file_name in os.listdir(self.data_folder) if file_name.endswith(".npy")]
        
    def evaluate(self,output_dir):
        with open(os.path.join(output_dir,"metrics.csv"), 'w', newline='') as csvfile:
            fieldnames = list(self.metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for file_name in self.data_files:
                file_dir = os.path.join(self.data_folder, file_name)
                data = np.load(file_dir)
                
                img = data[0,:,:,:4]
                seg = np.expand_dims(data[0,:,:,4], axis=(0,1))
                generated = data[0,:,:,5:]
                pred = np.expand_dims(np.sum((generated-img)**2, axis=2), axis=(0,1))
                
                metrics_img = {metric: metric_fn(seg,pred) for metric, metric_fn in self.metrics.items()}
                writer.writerow(metrics_img)
                
def evaluate_Brat(data_folder, output_dir):
    metrics = {
        "DICE_ET": partial(dice_specific, region_type="ET"),
        "DICE_TC": partial(dice_specific, region_type="TC"),
        "DICE_WT": partial(dice_specific, region_type="WT"),
        "AUROC": AUROC
    }
    
    evaluator = Brats_Evaluator(
        data_folder=data_folder,
        metrics=metrics
    )
    
    evaluator.evaluate(output_dir)
    
if __name__ == "__main__":
    evaluate_Brat('./output/anomaly_detection','./output/anomaly_detection')
    