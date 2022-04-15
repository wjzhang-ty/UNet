import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.DataLoading import MyDataset
from utils.utils import createNet, mask_to_image


# 超参数配置
from config import cfg


def predict_img(net, img, w, h):
    net.eval()

    with torch.no_grad():
        output = net(img)

        probs = F.softmax(output, dim=1)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((w, h)),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    return F.one_hot(full_mask.argmax(dim=0), cfg.n_classes).permute(2, 0, 1).numpy()

# # mask转图像。预测图为0、1、2，乘127.5得到0、128、255
# def mask_to_image(mask: np.ndarray):
#     return Image.fromarray((np.argmax(mask, axis=0) * 127.5).astype(np.uint8))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    in_file = 'img.jpg'
    out_file = 'output.jpg'
    net_name='unet'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建网络   
    net = createNet(net_name)
    net.to(device=device)
    net.load_state_dict(torch.load('./checkpoints/'+net_name+'.pth', map_location=device))

    # 加载图像
    img = MyDataset.preprocess(filename=in_file)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    full_img = Image.open(in_file)
    mask = predict_img(net=net, img=img, w=full_img.size[0] ,h=full_img.size[0])

    # 保存
    result = mask_to_image(mask)
    result.save(out_file)
