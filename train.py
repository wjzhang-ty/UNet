import torch
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from evaluate import evaluate
from tqdm import tqdm
from pathlib import Path
from utils.DataLoading import MyDataset
from utils.utils import createNet

# 自定义损失函数
from utils.dice_score import dice_loss
from utils.my_score import my_score

# 超参数配置
from config import cfg


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_name='unet'
    net = createNet(net_name).to(device=device)

    ###############
    ## 准备数据集 ##
    ###############
    train_set = MyDataset("./data/train/imgs", "./data/train/masks", "")
    val_set = MyDataset("./data/valid/imgs", "./data/valid/masks", "")
    
    # shuffle:随机打乱。
    # drop_last：丢掉batch_size分割后余下的数据。
    # pin_memory：数据直接保存在所内存中，速度快
    train_loader = DataLoader(train_set,shuffle=True,batch_size=cfg.batch_size,num_workers=4,pin_memory=True,)
    val_loader = DataLoader(val_set,shuffle=False,drop_last=True,batch_size=cfg.batch_size,num_workers=4,pin_memory=True)

    ###############
    ## optimizer ##
    ###############
    # optimizer = optim.RMSprop(net.parameters(), lr=cfg.learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
   
    # 两次dice loss不上升，下调学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
   
    # 混合精度
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 损失函数-他是个类在此处实例化
    criterion = nn.CrossEntropyLoss()

    ###############
    #### train ####
    ###############
    global_step = 0
    for epoch in range(cfg.epochs):
        net.train()
        with tqdm(total=len(train_set), desc=f"训练 {epoch + 1}/{cfg.epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"].to(device=device, dtype=torch.float32)
                true_masks = batch["mask"].to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=False):
                    masks_pred = net(images)
                    # 损失函数
                    loss = criterion(masks_pred, true_masks)
                    # loss += my_score(masks_pred)

                # 导数清零，后向传播什么的
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # 更新进度条
                pbar.update(images.shape[0])
                global_step += 1
                pbar.set_postfix(**{"loss": loss.item()})

        # 每轮epoch后都进行评估
        val_score = evaluate(net, val_loader, device)
        scheduler.step(val_score)
        print(val_score)

        # 每轮都保存一下训练好的参数
        Path("./checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save( net.state_dict(),'./checkpoints/'+net_name+'.pth')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    train()
