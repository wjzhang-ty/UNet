import torch
from evaluate import evaluate
from utils.utils import createNet
from utils.DataLoading import MyDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    net_name='transunet'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = createNet(net_name).to(device=device)
    net.load_state_dict(torch.load('./checkpoints/'+net_name+'.pth', map_location=device))

    test_set = MyDataset("./data/test/imgs", "./data/test/masks", "")
    val_loader = DataLoader(
        test_set,
        shuffle=False,  # 随机打乱
        drop_last=True,  # 丢掉batch_size分割后余下的数据
        batch_size=1,
        num_workers=4,
        pin_memory=True  # 数据直接保存在所内存中，速度快
    )

    val_score = evaluate(net, val_loader, device)
    print(val_score)