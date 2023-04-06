from torchvision.transforms import transforms as T
from PIL import Image
import numpy as np
import torch
import os
import csv
from guided_diffusion.utils import create_folders
from tqdm import tqdm
import nibabel as nib

from metrics import nonzero_masking

class image_processor():
    """
    basic pipeline of image loading, processing and saving
    """
    def __init__(
        self,
        image_size = [256,256],
        transforms = None,  #transforms apply to image. Default [ToTensor, Resize, Grayscale]
        load_method = None,  #ways of open image given a data path. Default Image.open 
        save_method = None  #ways of save image
    ):
        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Resize(image_size),
                T.Grayscale(),
            ])
        else:
            self.transforms = transforms
            
        if load_method is None:
            self.load_method = Image.open
        else:
            self.load_method = load_method
            
        if save_method is None:
            def save_np(output_dir, result):
                with open(output_dir, 'wb') as f:
                    np.save(f, result)
            self.save_method = save_np
        else:
            self.save_method = save_method
            
    def process(self, data_dir, output_dir):
        image = self.load_method(data_dir)
        processed_image = self.transforms(image)
        self.save_method(output_dir, processed_image)

            
def process_chexpert(data_path, label_dir, output_path, image_size = [256,256], transforms=None):
    processor = image_processor(image_size=image_size, transforms=transforms)
    
    for class_name in ["healthy", "pleural effusions"]:
        create_folders(os.path.join(output_path, class_name))
    
    #count total number of lines
    total_num = 0
    with open(label_dir, 'r') as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            total_num += 1
        
    def extract_output_name(img_dir):
        dirs = img_dir.split('/')
        name = "{}_{}.npy".format(dirs[2], dirs[3])
        return name
        
    def adapt_path(origin_path):
        dirs = origin_path.split('/')
        output_dir = os.path.join(*dirs)
        return output_dir

    
    with open(label_dir, 'r') as lf:
        reader = csv.DictReader(lf)
        reader = tqdm(reader, total=total_num, unit="pics")
        
        for row in reader:
            if row["Frontal/Lateral"] == "Lateral":
                continue
            if row["No Finding"] == "1.0":
                img_dir = os.path.join(data_path, adapt_path(row["Path"]))
                output_dir = os.path.join(output_path, "healthy", extract_output_name(row["Path"]))
                processor.process(img_dir, output_dir)                
            elif row["Pleural Effusion"] == "1.0":
                img_dir = os.path.join(data_path, adapt_path(row["Path"]))
                output_dir = os.path.join(output_path, "pleural effusions", extract_output_name(row["Path"]))
                processor.process(img_dir, output_dir)
    
    
def process_Brats2020(data_path, output_dir):
    
    for folder in ["images", "segmentations"]:
        for class_name in ['healthy', 'unhealthy']:
            create_folders(os.path.join(output_dir, folder, class_name))
    
    
    img_types = ("flair", "t1", "t1ce", "t2", "seg")
    
    global_counts = 0
    
    padding = (8,8,8,8)
    trans = T.Compose([
        T.ToTensor(),
        T.Pad(padding=padding)
    ])
    
    def load_method(data_path):
        data = nib.load(data_path)
        img = data.get_fdata()
        return img
    
    def save_method(output_dir, result):
        np.save(output_dir, result)
        
    def clamp_fn(image):
        mask = (image > 0) * 1.0
        image_fore = image[np.where(mask > 0)]
        min_val, max_val = np.percentile(image_fore, 1), np.percentile(image_fore, 99)
        image = np.clip(image, min_val, max_val)
        return image
        
            
    for i in tqdm(range(1, 370)):
        imgs = []
        imgs_folder = os.path.join(data_path, "BraTS20_Training_{:0>3d}".format(i))
        for img_type in img_types:
            img_path = os.path.join(imgs_folder, "BraTS20_Training_{:0>3d}_{}.nii.gz".format(i,img_type))
            img = load_method(img_path)
            imgs.append(img)
        
        #slice, process and save
        for slice in range(80, 130):        #only center slice (z in 80:-26)
            #processing segmentations
            seg = imgs[4][:,:,slice]
            seg = trans(seg)
            is_anomaly = seg.sum() > 0
            if  is_anomaly:
                seg_save_path = os.path.join(output_dir, "segmentations",'unhealthy', "BraTS20_Training_{:0>5d}_seg".format(global_counts))
            else:
                seg_save_path = os.path.join(output_dir, "segmentations",'healthy', "BraTS20_Training_{:0>5d}_seg".format(global_counts))
            save_method(seg_save_path, seg)
            
            #processing images
            result = []
            for j in range(4):
                img = imgs[j][:,:,slice]
                
                #clamp foreground to (1,99) percentiles
                img = clamp_fn(img)
                #normalize to (0,1)
                img = (img - img.min()) / (img.max() - img.min())
                
                img = trans(img)
                result.append(img)
            
            result = torch.cat(result, dim=0)
            if is_anomaly:
                result_save_path = os.path.join(output_dir, "images", 'unhealthy',"BraTS20_Training_{:0>5d}_image".format(global_counts))
            else:
                result_save_path = os.path.join(output_dir, "images", 'healthy',"BraTS20_Training_{:0>5d}_image".format(global_counts))
            save_method(result_save_path, result)
            
            global_counts += 1    
        
def seperate_dataset(data_dir, output_dir):
    import random        
    def copyFile(fileDir, tarDirs):
        pathDir = os.listdir(fileDir[0])
        num = len(pathDir)
        print(fileDir,num)

        random.shuffle(pathDir)
        
        samples = {}
        cnt = 0
        for tarDir, info in tarDirs.items():
            sample_num = int(info[0] * num)
            samples[tarDir] = pathDir[cnt:cnt+sample_num]
            cnt += sample_num
        #make sure all data points are covered
        if cnt < len(pathDir):
            samples[tarDir] += pathDir[cnt:]  
        
        for tarDir, sample in samples.items():
            for name in sample:
                saveDir = os.path.join(tarDir, name)
                data = np.load(os.path.join(fileDir[0], name))
                np.save(saveDir, data)
                saveDir = os.path.join(tarDirs[tarDir][1], _find_seg(name))
                data = np.load(os.path.join(fileDir[1], _find_seg(name)))
                np.save(saveDir, data)
                                            
    def _find_seg(image_file):
        seg_suffix = "seg.npy"

        image_split = image_file.split("_")
        image_split[-1] = seg_suffix
        seg_finded = "_".join(image_split)
        
        return seg_finded    


    folders = ['images','segmentations']
    class_names = ['healthy', 'unhealthy']
    
    for class_name in class_names:
        fileDir = (os.path.join(data_dir,folders[0], class_name),os.path.join(data_dir,folders[1], class_name))
            
        tarDirs = {
        os.path.join(output_dir,'train',folders[0], class_name):(0.8,os.path.join(output_dir,'train',folders[1], class_name)),
        os.path.join(output_dir, 'val', folders[0], class_name):(0.1,os.path.join(output_dir, 'val', folders[1], class_name)),
        os.path.join(output_dir, 'test', folders[0], class_name):(0.1,os.path.join(output_dir, 'test', folders[1], class_name))
        }
        
        for dir in [v[1] for v in tarDirs.values()] + list(tarDirs.keys()) :
            if not os.path.exists(dir):
                os.makedirs(dir)
        copyFile(fileDir, tarDirs) #rate means the percentage of images move to target1       

if __name__ == "__main__":
    # data_path = '..'
    # label_dir = os.path.join(data_path, "CheXpert-v1.0","train.csv")
    # output_dir = os.path.join(data_path, "CheXpert_Processed_1", "train")
    # process_chexpert(data_path, label_dir, output_dir)
    # data_path = '/Volumes/lxh_data/Brats2020/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    # output_dir = '/Volumes/lxh_data/Brats2020/Brats_Processed_Clean'
    # process_Brats2020(data_path, output_dir)
    data_path = '/Volumes/lxh_data/Brats2020/Brats_Processed_Clean'
    output_dir = '/Volumes/lxh_data/Brats2020/Brats_Processed_Clean_Split'
    seperate_dataset(data_path, output_dir)

                