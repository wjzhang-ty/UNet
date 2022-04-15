import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff

# 超参数配置
from config import cfg


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    # for batch in tqdm(dataloader, total=num_val_batches, desc='验证', unit='batch', leave=False):
    with tqdm(total=num_val_batches,  desc='验证', unit='batch', leave=False) as pbar:
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
            # 放进GPU
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, cfg.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # 预测mask
                mask_pred = net(image)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), cfg.n_classes).permute(0, 3, 1, 2).float()
                # 计算dice
                score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                pbar.update(image.shape[0])
                pbar.set_postfix(**{"dice": score.item()})
                dice_score+=score

    # 返回损失，分母不能为0
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
