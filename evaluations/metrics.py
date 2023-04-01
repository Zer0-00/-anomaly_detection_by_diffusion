import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Union
import os
import csv
from functools import partial
from tqdm import tqdm

ImageClass = Union[torch.Tensor,np.ndarray]

def dice_coeff(
    targets:ImageClass, 
    images:ImageClass,
    mask_fn=lambda x:x,
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
        image = mask_fn(image)
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
        sum_kwargs = {"dim":1, "keepdim": True}
        assert images.shape[2:] == targets.shape[2:],\
            f"the input and target images should share the same shape(H,W) get image: {images.shape[2:]} and target: {targets.shape[2:]} "  
            
    else:
        #numpy
        assert images.shape[1:3] == targets.shape[1:3],\
            f"the input and target images should share the same shape(H,W), get image: {images.shape[1:3]} and target: {targets.shape[1:3]}"
        sum_kwargs = {"axis":3, "keepdims": True}
            
    mask = ((images > images.min() * 1.0).sum(**sum_kwargs) == 4) * 1.0
    
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
        self.mask_fn = lambda x: x if mask_fn is None else mask_fn
        
        self.data_files = [file_name for file_name in os.listdir(self.data_folder) if file_name.endswith(".npy")]
        
    def evaluate_images(self,output_dir, store_data=True, use_tqdm=False):
        
        if store_data:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            csvfile = open(os.path.join(output_dir,"metrics.csv"), 'w', newline='')
            fieldnames = ['file_name']+list(self.metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
                
        metrics_imgs = {metric:[] for metric in self.metrics}
        
        iterf = tqdm(self.data_files) if  use_tqdm else self.data_files
              
        for file_name in iterf:
            file_dir = os.path.join(self.data_folder, file_name)
            data = np.load(file_dir)
            
            img = data[np.newaxis,0,:,:,:4]*1.0/255.0
            seg = np.expand_dims(data[0,:,:,4], axis=(0,-1))
            generated = data[np.newaxis,0,:,:,5:]*1.0/255.0
            pred = np.expand_dims(np.sum(np.sqrt((generated-img)**2), axis=3))
            pred = nonzero_masking(img, pred)
            thresh, _ = self.mask_fn(pred)
            
            metrics_img = {metric: metric_fn(seg,pred) for metric, metric_fn in self.metrics.items()}
            metrics_img["file_name"] = file_name
            metrics_img['threshold'] = thresh
            
            #filter the images that have no anomalies
            if metrics_img["AUROC_WT"] == -1:
                continue
            
            if store_data:
                writer.writerow(metrics_img)
            
            for key in metrics_imgs:
                metrics_imgs[key].append(metrics_img[key])
        
        if store_data:
            csvfile.close()
                        
        metrics = {k:(np.mean(v), np.std(v)) for k,v in metrics_imgs.items()}
        return metrics
                
    def evaluate(self,seg, pred, extra_data=None):
        metrics = {metric:metrics_fn(seg,pred) for metric,metrics_fn in self.metrics.items()}
        if extra_data is not None:
            metrics.update(extra_data)
            
        return metrics
                
def evaluate_Brat_images(data_folder, output_dir):
    """evaluate prediction generated from images from Brats dataset

    Args:
        data_folder: generated results folder
        output_dir: output directory
    """
    import cv2
    import pandas as pd
    
    def mask_fn(pred):
        thresh ,mask = cv2.threshold(pred, 0, 1, cv2.THRESH_OTSU)
        return thresh, mask
    
    def mask_dice(pred):
        _, mask = cv2.threshold(pred, 0, 1, cv2.THRESH_OTSU)
        return mask
    metrics = {
        #"DICE_ET": partial(region_specific_metrics, func=dice_coeff, region_type="ET"),
        #"DICE_TC": partial(region_specific_metrics, func=dice_coeff, region_type="TC"),
        "DICE_WT": partial(region_specific_metrics, func=dice_coeff, region_type="WT", mask_fn=mask_dice),
        #"AUROC_ET": partial(region_specific_metrics, func=AUROC, region_type="ET"),
        #"AUROC_TC": partial(region_specific_metrics, func=AUROC, region_type="TC"),
        "AUROC_WT": partial(region_specific_metrics, func=AUROC, region_type="WT"),
    }
    
    evaluator = BratsEvaluator(
        data_folder=data_folder,
        metrics=metrics,
        mask_fn=mask_fn
    )
    metrics_thresh = evaluator.evaluate_images(output_dir, tqdm=True)
    df = pd.DataFrame(metrics_thresh)
    output_path = os.path.join(output_dir, "total.csv")
    df.to_csv(output_path)
    
def calcu_best_thresh(data_folder, output_dir):
    import pandas as pd
    

    metrics_threshs = {'threshold': []}
    for k in metrics:
        metrics_threshs[k+'(Mean)'] = []
        metrics_threshs[k+'(Std)'] = []
    
    for threshold in tqdm(np.arange(0,0.5,0.005)):
        def mask_thresh(pred):
            return (pred >= threshold) * 1.0
        
        metrics = {
        #"DICE_ET": partial(region_specific_metrics, func=dice_coeff, region_type="ET"),
        #"DICE_TC": partial(region_specific_metrics, func=dice_coeff, region_type="TC"),
        "DICE_WT": partial(region_specific_metrics, func=dice_coeff, region_type="WT", mask_fn=mask_thresh),
        #"AUROC_ET": partial(region_specific_metrics, func=AUROC, region_type="ET"),
        #"AUROC_TC": partial(region_specific_metrics, func=AUROC, region_type="TC"),
        "AUROC_WT": partial(region_specific_metrics, func=AUROC, region_type="WT"),
        }
    
        evaluator = BratsEvaluator(
            data_folder=data_folder,
            metrics=metrics,
        )
    
        output_path = os.path.join(output_dir, f"{threshold}")
        metrics_thresh = evaluator.evaluate_images(output_path, store_data=False)
        metrics_threshs['threshold'].append(threshold)
        for k,v in metrics_thresh.items():
            metrics_threshs[k+'(Mean)'].append(v[0])
            metrics_threshs[k+'(Std)'].append(v[1])
            
    df = pd.DataFrame(metrics_threshs)
    output_path = os.path.join(output_dir, "total.csv")
    df.to_csv(output_path)
    
if __name__ == "__main__":
    #calcu_best_thresh('output/configs4/anomaly_detection','output/configs4/anomaly_detection')
    evaluate_Brat_images('output/configs4/anomaly_detection','output/configs4/anomaly_detection')