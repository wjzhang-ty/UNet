import numpy as np

def my_score(imgs,purpose=2):
    imgs = imgs.squeeze(0).cpu().detach().numpy()
    l=len(imgs)
    for patch in range(l+1):
        img=np.argmax(imgs[patch-1],axis=0)
        loss = 0.0
        for i in range(purpose):
            # 选取i分类的全部点
            zero = np.argwhere(img <= i)
            if len(zero) == 0: continue
            # 圆心
            x, y = sum(zero)//len(zero)

            # 位移
            list = zero-[x, y]
            xx = list[:, 0]
            yy = list[:, 1]
            angle = np.arctan2(yy, xx) * 180 / np.pi
            temp = np.argwhere(abs(angle+90) < 67.5)
            zero[temp, :] = zero[temp, :]+[0, 1]
            temp = np.argwhere(abs(angle-90) < 67.5)
            zero[temp, :] = zero[temp, :]-[0, 1]
            temp = np.argwhere(abs(angle) < 67.5)
            zero[temp, :] = zero[temp, :]-[1, 0]
            temp = np.argwhere(abs(angle) > 112)
            zero[temp, :] = zero[temp, :]+[1, 0]

            # 异常点求和取均值
            lx=zero[:,0]
            lx[lx<0]=0
            ly=zero[:,1]
            ly[ly<0]=0
            lossImg = img[lx,ly]
            lossImg=lossImg[lossImg>i]
            lossImg = lossImg-i
            loss += sum(lossImg)/len(zero)

    return loss/l