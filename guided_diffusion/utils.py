import json
import os
from collections import defaultdict
import torch
from torchvision import utils as vutils
import numpy as np
import csv

def load_parameters(cfg:str, default_dict:dict) -> dict:
    """
    loading configure json file.
    path of json file folder:./configs/
    """

    if cfg.isnumeric():
        para_name = 'configs{}.json'.format(cfg)
    elif cfg.endswith('.json'):
        para_name = cfg
    else:
        para_name = cfg + ".json"
    para_dir = os.path.join("./configs", para_name)
    cfgs_name = os.path.basename(para_dir)[:-5]
    print("configurations:"+cfgs_name)
    with open(para_dir, 'r') as f:
        args_dict = json.load(f)

    args = defaultdict(str)
    args.update(args_dict)
    
    #change type
    for k in default_dict:
        args[k] = type(default_dict[k])(args[k])
            
    args["cfgs_name"] = cfgs_name

    return args



def create_folders(f_dir):
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)


def save_images(images, images_folder_path):
    for images_name in images:
        imgs = images[images_name]
        
        imgs_folder = os.path.join(images_folder_path, images_name)
        create_folders(imgs_folder)
        
        for idx, img in enumerate(imgs):
            f_dir = os.path.join(imgs_folder, "{}.jpg".format(idx))
            vutils.save_image(img, f_dir)
            
def save_detail_metrics(metrics:dict, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = list(metrics.keys())
        writer.writerow(headers)
        #write rows
        for idx,_ in enumerate(metrics[headers[0]]):
            row = []
            for header in headers:
                row.append(float(metrics[header][idx]))
            writer.writerow(row)

def tensor2np(input_image:torch.Tensor, normalize=False):
    """
    change input tensor(C,H,W) to cv2 np.array(list(np.array(np.uint8)))
    """
    if normalize:
        input_image = (input_image - input_image.min() / input_image.max()-input_image.min())
        input_image = torch.permute(input_image, (1,2,0)).detach().cpu().numpy()
        input_image = (input_image * 255).astype(np.uint8)
        
    else:
        input_image = torch.permute(input_image, (1,2,0)).detach().cpu().numpy()
    return input_image

def normalize_image(input_images:torch.Tensor):
    """
    normalize batch image to (0,1) for every image in batch
    """
    pixel_dim = list(range(2, len(input_images.shape)))
    picture_shape = input_images.shape[2:]
    
    maxs, _ = torch.max(input_images.reshape(input_images.shape[0], input_images.shape[1], -1), dim=-1)
    maxs = maxs.reshape(*maxs.shape, *((1,)*len(picture_shape)))
    mins, _ = torch.min(input_images.reshape(input_images.shape[0], input_images.shape[1], -1), dim=-1)
    mins = mins.reshape(*mins.shape, *((1,)*len(picture_shape)))

    normalized_images = (input_images - mins.repeat(1,1,*picture_shape)) / (maxs-mins).repeat(1,1,*picture_shape)
    
    return normalized_images
