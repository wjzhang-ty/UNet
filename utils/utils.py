
import torch
import numpy as np
from PIL import Image
from config import cfg
from pathlib import Path

from TranUNet.transunet import TransUNet
from UNet.UNet import UNet

# 根据name创建网络并返回
def createNet(name='unet'):
    if name == 'unet':
        return UNet(cfg.in_cannel,cfg.n_classes)

    if name == 'transunet':
        return TransUNet(img_dim=cfg.transunet.img_dim,
                    in_channels=cfg.in_cannel,
                    out_channels=cfg.transunet.out_channels,
                    head_num=cfg.transunet.head_num,
                    mlp_dim=cfg.transunet.mlp_dim,
                    block_num=cfg.transunet.block_num,
                    patch_dim=cfg.transunet.patch_dim,
                    class_num=cfg.n_classes)

# mask转图像。预测图为0、1、2，乘127.5得到0、128、255
def mask_to_image(mask: np.ndarray):
    return Image.fromarray((np.argmax(mask, axis=0) * 127.5).astype(np.uint8))


# 裁剪粗分割的结果，返回裁剪起点（左上），横向长度，纵向长度
def cutting(channel_one):
    img = np.argwhere(channel_one >= 0.5)
    if len(img) <= 0: return 0,0,0,0
    center_x, center_y = sum(img)//len(img)
    max_x,max_y=np.amax(img, axis=0)
    min_x,min_y=np.amin(img, axis=0)
    range = cfg.my.trans_img_dim/2

    y=int(max(min(center_y-range,min_y),0))
    y_range=int(max(range*2,max_y-min_y))
    x=int(max(min(center_x-range,min_x),0))
    x_range=int(max(range*2,max_x-min_x))

    return y,x,y_range,x_range

# 保存checkpoints
def save_checkpoints(net,path: str,save_name: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save( net.state_dict(),path+save_name+'.pth')