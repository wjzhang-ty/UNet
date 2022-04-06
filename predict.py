import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.DataLoading import MyDataset


from UNet import UNet

def predict_img(net, full_img):
    net.eval()

    with torch.no_grad():
        output = net(img)

        probs = F.softmax(output, dim=1)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    return F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()

# 归一化的图片转原图
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    in_file = 'img.jpg'
    out_file = 'output.jpg'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # UNet，加载参数    
    net = UNet(3,2)
    net.to(device=device)
    net.load_state_dict(torch.load('MODEL.pth', map_location=device))

    # 加载图像
    img = MyDataset.preprocess(filename=in_file)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    mask = predict_img(net=net, full_img=Image.open(in_file))

    # 保存
    result = mask_to_image(mask)
    result.save(out_file)