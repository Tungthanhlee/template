import os
import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset
import cv2
cv2.setNumThreads(0)
# cv2.oc1.setUseOpenCL(False)
from albumentations import *
from albumentations.pytorch import ToTensor
from config import get_cfg_defaults

class Cifar(Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.df = pd.read_csv(os.path.join(cfg.DIRS.DATA, "folds", f"train_fold{cfg.TRAIN.FOLD}.csv"))
        self.data_root = os.path.join(cfg.DIRS.DATA, "train")
        size = cfg.DATA.IMG_SIZE

        self.transform = getTransform(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.loc[idx]
        img_path = os.path.join(self.data_root, info["id"])
        image = self.load_img(img_path)
        label = info["label"]

        return image, label
    
    def load_img(self, img_path):
        image = cv2.imread(img_path)
        image = self.transform(image = image)

        image = image["image"]
        return image


def getTransform(size):

    transforms_train = Compose([
        Resize(size, size),
        # OneOf([
        #     ShiftScaleRotate(
        #         shift_limit=0.0625,
        #         scale_limit=0.1,
        #         rotate_limit=30,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=0),
        #     GridDistortion(
        #         distort_limit=0.2,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=0),
        #     OpticalDistortion(
        #         distort_limit=0.2,
        #         shift_limit=0.15,
        #         border_mode=cv2.BORDER_CONSTANT,
        #         value=0),
        #     NoOp()
        # ]),
        # RandomSizedCrop(
        #     min_max_height=(int(size * 0.75), size),
        #     height=size,
        #     width=size,
        #     p=0.25),
        OneOf([
            RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4),
            RandomGamma(gamma_limit=(50, 150)),
            IAASharpen(),
            IAAEmboss(),
            CLAHE(clip_limit=2),
            NoOp()
        ]),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            MedianBlur(blur_limit=3),
            Blur(blur_limit=3),
        ], p=0.15),
        OneOf([
            RGBShift(),
            HueSaturationValue(),
        ], p=0.05),
        HorizontalFlip(p=0.5),
        Normalize(),
        ToTensor()
    ])
    return transforms_train

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    dts = Cifar(cfg)
    print(dts)
    # img, label = dts.__getitem__(idx=3)
    # print(img.shape,"\n", label)
    # print(dts.__len__())