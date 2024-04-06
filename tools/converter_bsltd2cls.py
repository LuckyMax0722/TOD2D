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

from lib.config import CONF

import multiprocessing
from multiprocessing import Manager

from tqdm import tqdm
from functools import partial


def visualization_image(image_path, obj_cat_list, obj_bbox_list):
    # 读取PNG图片
    image = cv2.imread(image_path)

    # 定义颜色（蓝色）
    color = (255, 0, 0)

    # 绘制边界框和标签
    for label, bbox in zip(obj_cat_list, obj_bbox_list):
        x_min, y_min, x_max, y_max = bbox
        # 绘制矩形
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        # 添加标签
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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


def get_bbox_image(bbox, image_path):
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
        '''
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        target_path = os.path.join(CONF.dataset_tlc_classifier.images, img_name + '_' + str(object_idx) + '.jpg')
        cv2.imwrite(target_path, im)
        '''

        return True, ''


def crete_datasets(image, label_counts):
    boxes = image['boxes']
    image_path = image['path']
    image_path = image_path.split("./")[1]
    image_path = os.path.join(CONF.PATH.DATA_BSLTD, image_path)

    # 获取标签
    img_path_list = []
    obj_cat_list = []
    obj_bbox_list = []

    for box in boxes:
        bbox = [int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])]

        state, target_path = get_bbox_image(bbox, image_path)

        if state:
            label = box['label']

            obj_cat_list.append(label)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        else:
            continue

        # obj_bbox_list.append(bbox)

    # visualization_image(image_path, obj_cat_list, obj_bbox_list)
    # Press 0 to kill the image window


def main(image, label_counts):
    boxes = image['boxes']
    if not boxes:
        return
    else:
        crete_datasets(image, label_counts)


if __name__ == '__main__':
    yaml_path = os.path.join(CONF.PATH.DATA_BSLTD, 'train.yaml')
    out_dir = os.path.join(CONF.PATH.DATASET_CLS, 'bsltd_cls')

    # 创建dataset路径并确保目录存在
    # os.makedirs(out_dir, exist_ok=True)

    images = yaml.full_load(open(yaml_path, 'rb').read())

    # 创建共享的 label_counts 字典
    manager = Manager()
    label_counts = manager.dict()

    # 获取处理器核心数量
    num_cores = multiprocessing.cpu_count()

    # 使用多进程处理数据集中的每张图片
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 创建进度条
        with tqdm(total=len(images), desc='Processing Images') as pbar:
            func = partial(main, label_counts=label_counts)
            for _ in pool.imap_unordered(func, images):
                pbar.update(1)

    print("所有图片处理完成")

    print(label_counts)
