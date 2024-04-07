#!/usr/bin/env python
"""
This script Converts Yaml annotations to Pascal .xml Files
of the Bosch Small Traffic Lights Dataset.
Example usage:
    python bosch_to_pascal.py input_yaml out_folder
"""

import sys
import yaml
import os
import cv2
import multiprocessing
from functools import partial
from tqdm import tqdm

from lib.config import CONF


def visualization_image(image_path, obj_bbox_list):
    # 读取PNG图片
    image = cv2.imread(image_path)

    # 定义颜色（蓝色）
    color = (255, 0, 0)

    # 绘制边界框和标签
    for bbox in obj_bbox_list:
        x_min, y_min, x_max, y_max = bbox
        # 绘制矩形
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        '''
        # 添加标签
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        '''

    # 显示图片
    cv2.imshow('Test Image', image)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口


def get_resize_object_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox

    w = x_max - x_min
    h = y_max - y_min

    x_c = x_min + w / 2
    y_c = y_min + h / 2

    # rectangle to square
    b_length = max(w, h)

    x_min = int(x_c - b_length / 2 - 3)
    x_max = int(x_c + b_length / 2 + 3)
    y_min = int(y_c - b_length / 2 - 3)
    y_max = int(y_c + b_length / 2 + 3)

    return [x_min, y_min, x_max, y_max]


def get_bbox_image(object_idx, bbox, image_path):
    resize_bbox = get_resize_object_bbox(bbox)

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 判断bbox是否能够截取到合理的图像
    if resize_bbox[0] < 0 or resize_bbox[1] < 0 or resize_bbox[2] > img.shape[1] or resize_bbox[3] > img.shape[0]:
        return False, ''
    else:
        cutout = img[resize_bbox[1]:resize_bbox[3], resize_bbox[0]:resize_bbox[2]]
        im = cv2.resize(cutout, (224, 224))  # BGR

        # visualization
        '''
        cv2.imshow('Image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        # 保存JPG图片
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        target_path = os.path.join(CONF.dataset_tlc_classifier.images, img_name + '_' + str(object_idx) + '.jpg')
        cv2.imwrite(target_path, im)

        return True, target_path


def get_bbox_label(label):
    # return color, shape
    if label == 'Red':
        return 0, 0
    elif label == 'RedLeft':
        return 0, 1
    elif label == 'RedRight':
        return 0, 2
    elif label == 'RedStraight':
        return 0, 3
    elif label == 'RedStraightLeft':
        return 0, 4
    elif label == 'Yellow':
        return 1, 4
    elif label == 'Green':
        return 2, 0
    elif label == 'GreenLeft':
        return 2, 1
    elif label == 'GreenRight':
        return 2, 2
    elif label == 'GreenStraight':
        return 2, 3
    elif label == 'GreenStraightRight':
        return 2, 4
    elif label == 'GreenStraightLeft':
        return 2, 4
    elif label == 'off':
        return 3, 4


def crete_datasets(image, mode):
    boxes = image['boxes']
    image_path = image['path']
    if mode == 'train':
        image_path = image_path.split("./")[1]
        image_path = os.path.join(CONF.PATH.DATA_BSLTD, image_path)
    elif mode == 'test':
        image_path = image_path.split('/')[-1]
        image_path = os.path.join(CONF.PATH.DATA_BSLTD, 'rgb/test', image_path)
    else:
        raise ValueError("Mode is wrong")

    # 获取标签
    img_path_list = []
    obj_col_list = []
    obj_cls_list = []
    obj_bbox_list = []

    for object_idx, box in enumerate(boxes):
        bbox = [int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])]

        # Add bbox for Vis
        obj_bbox_list.append(bbox)

        state, target_path = get_bbox_image(object_idx, bbox, image_path)

        if state:
            # Saved Jpg Path
            img_path_list.append(target_path)

            # Save label
            ## Colour and Class
            label = box['label']
            col, cls = get_bbox_label(label)
            obj_col_list.append(col)
            obj_cls_list.append(cls)


        else:
            continue

    # visualization_image(image_path, obj_bbox_list)
    # Press 0 to kill the image window

    # txt文件的创建
    txt_output_path = CONF.dataset_tlc_classifier.labels_txt_path

    # 打开一个 TXT 文件进行追加写入
    with open(txt_output_path, 'a') as f:
        # 遍历图片信息列表，并将每一行数据追加到文件末尾
        for path, col, cls in zip(img_path_list, obj_col_list, obj_cls_list):
            line = f"{path} {col} {cls}\n"
            f.write(line)


def main(image, mode):
    boxes = image['boxes']
    if not boxes:
        return
    else:
        crete_datasets(image, mode)


if __name__ == '__main__':
    # Part 1 Images
    train_yaml_path = os.path.join(CONF.PATH.DATA_BSLTD, 'train.yaml')

    # 创建dataset路径并确保目录存在
    os.makedirs(CONF.dataset_tlc_classifier.images, exist_ok=True)
    os.makedirs(CONF.dataset_tlc_classifier.labels, exist_ok=True)

    images = yaml.full_load(open(train_yaml_path, 'rb').read())

    # 获取处理器核心数量
    num_cores = multiprocessing.cpu_count()

    # 使用多进程处理数据集中的每张图片
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 创建进度条
        with tqdm(total=len(images), desc='Processing BSLTD Images Part 1') as pbar:
            func = partial(main, mode='train')
            for _ in pool.imap_unordered(func, images):
                pbar.update(1)

    print("Finish Processing BSLTD Images Part 1")

    # Part 2 Images
    test_yaml_path = os.path.join(CONF.PATH.DATA_BSLTD, 'test.yaml')

    images = yaml.full_load(open(test_yaml_path, 'rb').read())

    # 获取处理器核心数量
    num_cores = multiprocessing.cpu_count()

    # 使用多进程处理数据集中的每张图片
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 创建进度条
        with tqdm(total=len(images), desc='Processing BSLTD Images Part 2') as pbar:
            func = partial(main, mode='test')
            for _ in pool.imap_unordered(func, images):
                pbar.update(1)

    print("Finish Processing BSLTD Images Part 2")

