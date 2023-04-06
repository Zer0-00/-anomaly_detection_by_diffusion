from torch.utils.data import Dataset,DataLoader
import os
from torchvision import transforms as T
import torch
import skimage.io as io
import numpy as np

def load_data(
    *,
    data_dir,
    batch_size,
    dataset,
    deterministic=False,
    class_labels=True,
    test=False,
    limited_num=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param test: if True, yield results from test set
    :param limited_num: if None, yield unlimited number of samples, else yield limited_num * batch_size samples, if <= 0, yield len(Dataset) sanmples
    """
    
    if dataset.lower() == "brats2020":
        dataset = Brats2020(
            data_dir=data_dir,
            test=test,
            class_labels=class_labels
        )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    if limited_num is None:
        while True:
            yield from loader
    elif limited_num <= 0:
            yield from loader
    else:
        loader = iter(loader)
        for _ in range(limited_num):
            data = next(loader)
            yield data




class MVtec_Leather(Dataset):
    def __init__(self, data_dir,
                 anomalous = False, 
                 img_size = (256,256),
                 rgb = True,
                 prepare = ("random_crop"),
                 include_good = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.anomalous = anomalous
        self.rgb = rgb
        self.img_size = img_size
        self.prepare = prepare
        self.include_good = include_good
        self.classes = ["color", "cut", "fold", "glue", "poke"]
        
        if not self.anomalous:
            self.data_dir = os.path.join(self.data_dir, "train", "good")
        
        #set transformation
        transform_list = [T.ToPILImage()]
        
        if not self.rgb:
            self.channel = 1
            transform_list.append(T.Grayscale(num_output_channels=self.channel))
        else:
            self.channel = 3
        
        transform_list.append(T.ToTensor())
        

        
        # normalize_factor = ((0.5,)*self.channel, (0.5,)*self.channel)
        # transform_list.append(T.Normalize(*normalize_factor))
        
        self.transform = T.Compose(transform_list)
        
        
        prepare_list = []
        if "random_crop" in self.prepare:
            prepare_list.append(T.RandomCrop(self.img_size))
        else:
            prepare_list.append(T.Resize(self.img_size))
        
        if "random_rotation" in self.prepare:
            prepare_list.append(T.RandomRotation(180))
            
        self.preparation = T.Compose(prepare_list)
            
        #process filenames
        if self.include_good:
            self.classes.append("good")
        
        if self.anomalous:
            mask_transform_list = [
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
            ]
            self.mask_transform = T.Compose(mask_transform_list)
            self.filenames = []
            for cl in self.classes:
                class_dir = os.path.join(self.data_dir, "test", cl) 
                self.filenames += [os.path.join(class_dir,file_name) for file_name in os.listdir(class_dir) if file_name.endswith(".png")]
        else:
            self.filenames = [os.path.join(self.data_dir,file_name) for file_name in os.listdir(self.data_dir) if file_name.endswith(".png")]
        
        self.filenames = sorted(self.filenames, key = lambda x: int(x[-7:-4]))    
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        output = {"filename":self.filenames[idx]}
        
        img = io.imread(self.filenames[idx])
        
        #get mask if needed
        if self.anomalous:
            name_split = self.filenames[idx].split(os.sep)
            if name_split[-2] == "good":
                mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
            else:
                mask = io.imread(os.path.join(self.data_dir, "ground_truth", name_split[-2], name_split[-1][:-4]+ "_mask.png"))

            mask = self.mask_transform(mask)
            
        #prepare image and mask        
        img = self.transform(img)
        
        
        
        #pre-prepare image and mask
        if self.anomalous:
            cat_img = torch.cat((img, mask), 0)
            prepared_img = self.preparation(cat_img)
            
            output["input"] = prepared_img[:self.channel]
            output["mask"] = prepared_img[self.channel:]
        else:
            prepared_img = self.preparation(img)
            output["input"] = prepared_img
        
        return output

class CheXpert(Dataset):
    def __init__(
        self,
        data_dir,
        anomalous = False
    ):  
        super().__init__()
        self.data_dir = data_dir
        self.anomalous = anomalous
        
        self.health_path = os.path.join(data_dir,"healthy")
        self.image_dirs = [os.path.join(self.health_path, image_dir) for image_dir in os.listdir(self.health_path)]
        
        if self.anomalous:
            self.y = [0]*len(self.image_dirs)
            self.anomaly_path = os.path.join(data_dir,"pleural_effusions")
            self.image_dirs += [os.path.join(self.anomaly_path, image_dir) for image_dir in os.listdir(self.anomaly_path)]
            self.y += [1]*len(self.image_dirs)
            
        self.transforms = T.Compose([])
        
    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        
        #the dataset has been changed to (C,H,W)
        image = torch.Tensor(np.load(image_dir))
        
        image = self.transforms(image)
        
        output = {
            "input": image,
            "name" : image_dir
        }
        
        if self.anomalous:
            output["y"] = self.y[idx]
        
        return output

class Brats2020(Dataset):
    def __init__(
        self,
        data_dir,
        test=False,
        trans=None,
        seg_trans=None,
        class_labels=True,
        classes_included='both',
    ):
        """Dataset for Brats-2020 dataset.

        Args:
            data_dir: data directory
            test (bool, optional): whether generating the test dataset(i.e. including segmentations). Defaults to False.
            trans (optional): Preprocess functions for images. Defaults to None.
            seg_trans (optional): Preprocess functions for segmentations. Defaults to None.
            class_labels (bool, optional): Whether to generate the class labels. Defaults to True.
            classes_included (str, optional): the classes included in dataset. Choose from ['normal','anomaly','both'].Defaults to 'both'.
        """
        super().__init__()
        
        assert classes_included in ['normal','anomaly', 'both'], "Class labels should be set to 'normal', 'anomaly' or 'both'."
        
        self.data_dir = data_dir
        self.test = test
        self.class_labels = class_labels
        self.classes_included = classes_included
        self.class_names = ['healthy', 'unhealthy']
        self.process_seg = self.test or self.class_labels
        
        self.image_folder = os.path.join(self.data_dir, "images")
        if self.process_seg:
            self.segmentation_folder = os.path.join(self.data_dir, "segmentations")
        
        if self.classes_included == 'anomaly':
            image_files = os.listdir(os.path.join(self.image_folder, self.class_names[1]))
            self.image_dirs = [os.path.join(self.image_folder, self.class_names[1], image_file) for image_file in image_files]
            if self.process_seg:
                self.segmentation_files = [[os.path.join(self.segmentation_folder, self.class_names[1],self._find_seg(image_file)) for image_file in image_files]]
        else:
            image_files = os.listdir(os.path.join(self.image_folder, self.class_names[0]))
            self.image_dirs = [os.path.join(self.image_folder, self.class_names[0], image_file) for image_file in image_files]
            if self.process_seg:
                self.segmentation_dirs = [os.path.join(self.segmentation_folder, self.class_names[0],self._find_seg(image_file)) for image_file in image_files]
                
            if self.classes_included == 'both':
                image_files = os.listdir(os.path.join(self.image_folder, self.class_names[1]))
                image_dirs = [os.path.join(self.image_folder, self.class_names[1], image_file) for image_file in image_files]
                self.image_dirs += image_dirs
                if self.process_seg:
                    segmentation_dirs = [os.path.join(self.segmentation_folder, self.class_names[1],self._find_seg(image_file)) for image_file in image_files]
                    self.segmentation_dirs += segmentation_dirs
        
        if trans is None:
            self.transforms = T.Compose([])
        else:
            self.transforms = trans
            
        if seg_trans is None:
            self.seg_transforms = T.Compose([])
        else:
            self.seg_transforms = seg_trans
            
            
    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, idx) -> dict:
        outputs = dict()
        
        image_dir = self.image_dirs[idx]
        image = torch.Tensor(self._load_method(image_dir))
        image = self.transforms(image)
        #outputs["input"] = image
        
        #matching segmentation to image

        if self.process_seg:
            seg_dir = self.segmentation_dirs[idx]
            seg = torch.Tensor(self._load_method(seg_dir))
            seg = self.seg_transforms(seg)
        
        #determine whether is normal (0:normal, 1: abnormal)
        if self.class_labels:
            y = 1 if(seg.max() > 0) else 0

            outputs["y"] = y
        if self.test:
            #generating segmentation ground truth
            outputs["seg"] = seg
            
        return image, outputs
        
    def _load_method(self, image_dir):
        return np.load(image_dir)
    
    def _find_seg(self, image_file):
        seg_suffix = "seg.npy"

        image_split = image_file.split("_")
        image_split[-1] = seg_suffix
        seg_finded = "_".join(image_split)
        
        return seg_finded           