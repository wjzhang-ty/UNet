import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from UNet import UNet
from evaluate import evaluate
from tqdm import tqdm
from pathlib import Path
from utils.DataLoading import MyDataset
from utils.dice_score import dice_loss


def train(epochs,batch_size,in_cannel,n_classes):
    learning_rate=1e-5

    gradScaler = True # 是否启用混合精度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(in_cannel,n_classes)
    net.to(device=device)

    ###############
    ## 准备训练集 ##
    ###############
    path = "."
    dataset = MyDataset(path + "/data/imgs", path + "/data/masks", "")
    val_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - val_len
    # 分割训练集、测试集
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(0)
    )

    ###############
    ### Loader ####
    ###############
    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,  # 随机打乱
        drop_last=True,  # 丢掉batch_size分割后余下的数据
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,  # 数据直接保存在所内存中，速度快
    )

    ###############
    ## optimizer ##
    ###############
    optimizer = optim.RMSprop(
        net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9
    )
    # 两次dice loss不上升，下调学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    # 混合精度
    grad_scaler = torch.cuda.amp.GradScaler(enabled=gradScaler)
    # 损失函数-他是个类在此处实例化
    criterion = nn.CrossEntropyLoss()
    # criterion = F.mse_loss()

    ###############
    #### train ####
    ###############
    global_step = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=train_len, desc=f"进度 {epoch + 1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"].to(device=device, dtype=torch.float32)
                true_masks = batch["mask"].to(device=device, dtype=torch.long)
                x=torch.max(true_masks)
                # true_masks=F.one_hot(true_masks).permute(0, 3, 1, 2)

                with torch.cuda.amp.autocast(enabled=gradScaler):
                    masks_pred = net(images)
                    # 交叉熵+筛子系数
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item()

                # 更新进度条
                pbar.update(images.shape[0])
                global_step += 1
                pbar.set_postfix(**{"loss": loss.item()})

                # 评估
                division_step = train_len // (10 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)
                        print(val_score)

        # 每轮都保存一下训练好的参数
        Path(path + "/checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            net.state_dict(),
            str((path + "/checkpoints/checkpoint_epoch{}.pth").format(epoch + 1)),
        )


if __name__ == "__main__":
    ###############
    #### 超参数 ###
    ###############
    epochs = 5
    batch_size = 4
    in_cannel = 3 # 输入图像的通道数
    n_classes = 3 # 输出通道（分几类）

    
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    train(epochs,batch_size,in_cannel,n_classes)
