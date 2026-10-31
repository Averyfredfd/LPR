# -*-coding: utf-8 -*-
import os
import cv2
import random
import shutil
import numpy as np
from tqdm import tqdm
from pybaseutils import file_utils, image_utils


def get_plate_licenses(plate):
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
                 "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    try:
        result = [provinces[int(plate[0])], alphabets[int(plate[1])]]
        result += [ads[int(p)] for p in plate[2:]]
        return "".join(result)
    except Exception:
        return "unknown"


def parser_annotations(image_file):
    filename = os.path.basename(image_file)
    try:
        annotations = filename.split("-")
        box = annotations[2].replace("&", "_").split("_")
        plate = annotations[4].split("_")
        plate = get_plate_licenses(plate)
        box = [int(b) for b in box]

        if box[2] <= box[0] or box[3] <= box[1]:
            return None

        return {"filename": filename, "bbox": box, "plate": plate}
    except Exception:
        return None


def convert_ccpd_to_yolo_and_crops(ccpd_root_dir, out_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    遍历 CCPD 根目录下的所有子文件夹，汇总转换
    """
    print(f"正在扫描 CCPD 根目录: {ccpd_root_dir}")
    all_image_list = []

    # 1. 遍历所有子文件夹
    for item in os.listdir(ccpd_root_dir):
        sub_dir = os.path.join(ccpd_root_dir, item)
        if os.path.isdir(sub_dir):
            print(f"扫描子文件夹: {item}")
            # 获取该子文件夹下的所有图片
            images = file_utils.get_images_list(sub_dir)
            all_image_list.extend(images)

    # 如果根目录下直接有图片，也可以加上（根据实际目录结构调整）
    root_images = file_utils.get_images_list(ccpd_root_dir, recursive=False)
    if root_images:
        all_image_list.extend(root_images)

    # 去重（防止软链接或重复扫描）
    all_image_list = list(set(all_image_list))
    print(f"✅ 总共发现图片: {len(all_image_list)} 张")

    # 2. 打乱并划分
    random.seed(42)
    random.shuffle(all_image_list)

    total = len(all_image_list)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    splits = {
        "train": all_image_list[:train_end],
        "val": all_image_list[train_end:val_end],
        "test": all_image_list[val_end:]
    }

    # 3. 创建输出目录
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "crops", split), exist_ok=True)

    # 4. 处理数据
    print("开始转换并保存文件...")
    for split_name, img_paths in splits.items():
        for img_path in tqdm(img_paths, desc=f"处理 {split_name}"):
            info = parser_annotations(img_path)
            if not info:
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            xmin, ymin, xmax, ymax = info["bbox"]
            plate_str = info["plate"]
            filename = info["filename"]
            base_name = os.path.splitext(filename)[0]

            # --- 生成 YOLO 标签 ---
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            # 写入 txt
            label_path = os.path.join(out_dir, "labels", split_name, f"{base_name}.txt")
            with open(label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

            # 复制原图
            dst_img_path = os.path.join(out_dir, "images", split_name, filename)
            shutil.copy(img_path, dst_img_path)

            # --- 裁剪车牌并保存 ---
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            if xmax > xmin and ymax > ymin:
                crop_img = image[ymin:ymax, xmin:xmax]
                # 命名：车牌号_原文件名.jpg
                crop_filename = f"{plate_str}_{base_name}.jpg"
                crop_save_path = os.path.join(out_dir, "crops", split_name, crop_filename)
                cv2.imwrite(crop_save_path, crop_img)

    print(f"\n✅ 所有转换完成！输出目录: {out_dir}")
    print(f"训练集: {len(splits['train'])} 张")
    print(f"验证集: {len(splits['val'])} 张")
    print(f"测试集: {len(splits['test'])} 张")


if __name__ == "__main__":
    # CCPD 根目录（里面包含 ccpd_base, ccpd_blur, ccpd_green 等子文件夹）
    ccpd_root_dir = "./dataset/CCPD2020"

    # 输出目录
    output_dir = ("./ccpd_LPRNet_dataset")

    convert_ccpd_to_yolo_and_crops(ccpd_root_dir, output_dir, split_ratio=(0.8, 0.1, 0.1))