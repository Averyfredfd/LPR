import numpy as np
import cv2
import torch
from lprr.LPRNet import CHARS, build_lprnet


def transform(img):
    img = img.astype('float32')
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


# [核心修改 1]：抽离初始化过程。保证模型只在系统启动时被硬盘读取一次，杜绝 I/O 阻塞。
def init_lprnet(weight_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 修正 phase=False，关闭模型中 Dropout 等训练机制
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
    # 增加 map_location 确保设备匹配
    lprnet.load_state_dict(torch.load(weight_path, map_location=device))
    lprnet.to(device)
    lprnet.eval()  # 必须开启评估模式
    return lprnet, device


# [核心修改 2]：接收全局传入的模型与设备，而不是自己去硬盘读
def de_lpr(coord, im0, lprnet, device):
    img = im0[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    if img.size == 0:
        return np.array([])  # 防御性编程：防止越界截取导致 opencv 崩溃

    im = cv2.resize(img, (94, 24))
    im = transform(im)

    # [核心修改 3]：解决 Creating a tensor from a list of numpy.ndarrays is extremely slow 警告
    ims = torch.from_numpy(np.array([im])).float()

    # [核心修改 4]：切断计算图，防止显存泄漏，极大加速推理
    with torch.no_grad():
        prebs = lprnet(ims.to(device))  # classifier prediction

    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]  # 对每张图片 [68, 18]
        preb_label = list()
        for j in range(preb.shape[1]):  # 18
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:  # 记录重复字符
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # 去除重复字符和空白字符'-'
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    plat_num = np.array(preb_labels)
    return plat_num


def dr_plate(im0, coord, plat_num):
    if plat_num.size == 0:
        return
    x1 = int(coord[0])
    x2 = int(coord[1])
    plate = np.array(plat_num)
    a = ""
    for i in range(0, plate.shape[1]):
        b = CHARS[plate[0][i]]
        a += b

    cv2.putText(im0, a, (x1, x2), 0, 1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)