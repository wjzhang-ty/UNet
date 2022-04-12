import torch
import numpy as np
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import splitext


class MyDataset:
    """
    images_dir：训练集地址。
    masks_dir：训练集的标注地址。
    mask_suffix：标注名称后缀。
    """

    def __init__(self, images_dir, masks_dir, mask_suffix="") -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if not file.startswith(".")
        ]

    def __getitem__(self, idx):
        # 组装path
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        # 打开图片
        mask = self.preprocess(filename=mask_file[0], is_mask=True)
        img = self.preprocess(filename=img_file[0])

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(filename, is_mask=False):
        """
        图片处理
        filename：图片路径+名字
        is_mask；是否为标注图片 Bollean
        """
        # 打开文件
        img = Image.open(filename)

        # 宽高对齐，保证上采样、跨层链接维度对齐
        img = img.resize((640, 432), Image.NEAREST)
        
        # 转格式
        img = np.asarray(img)

        # mask把颜色重置为0、1、2
        # 训练集归一化。
        if is_mask:
            img[img == 128]=1
            img[img == 255]=2
        else:
            img = (img / 255).transpose((2, 0, 1))

        return img