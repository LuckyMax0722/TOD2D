from __future__ import print_function

import sys
from pathlib import Path

import cv2
import os
import random
import multiprocessing
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.dtld_parsing.driveu_dataset import DriveuDatabase
from lib.config import CONF


def visualize(file_path, objects):
    # 读取图像
    image = cv2.imread(file_path)

    for o in objects:
        cv2.rectangle(
            image,
            (o.x, o.y),
            (o.x + o.width, o.y + o.height),
            o.color_from_attributes(),
            1,
        )

    # 显示带有边界框的图像
    cv2.imshow("Image with bounding box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_database(label_file, data_base_dir):
    # 读取数据集
    database = DriveuDatabase(label_file)
    if not database.open(data_base_dir):
        return False

    return database


def convert_to_yolo_label(x_min, y_min, w, h, image_width, image_height):
    x_center = (x_min + (w / 2)) / image_width
    y_center = (y_min + (h / 2)) / image_height
    width = w / image_width
    height = h / image_height

    # Round values to keep only 6 decimal places
    x_center = round(x_center, 6)
    y_center = round(y_center, 6)
    width = round(width, 6)
    height = round(height, 6)

    return x_center, y_center, width, height


def get_object_bbox(object, image_shape):
    return convert_to_yolo_label(object.x, object.y, object.width, object.height, image_shape[1], image_shape[0])


def assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base):
    attributes = object.attributes
    if attributes['state'] == 'off':
        obj_cat_list.append(cls_base)  # class: cls_base
        normalized_obj_bbox_list.append(get_object_bbox(object, image.shape))
    elif attributes['state'] == 'red':
        obj_cat_list.append(cls_base + 1)  # class: cls_base + 1
        normalized_obj_bbox_list.append(get_object_bbox(object, image.shape))
    elif attributes['state'] == 'yellow':
        obj_cat_list.append(cls_base + 2)  # class: cls_base + 2
        normalized_obj_bbox_list.append(get_object_bbox(object, image.shape))
    elif attributes['state'] == 'red_yellow':
        obj_cat_list.append(cls_base + 3)  # class: cls_base + 3
        normalized_obj_bbox_list.append(get_object_bbox(object, image.shape))
    elif attributes['state'] == 'green':
        obj_cat_list.append(cls_base + 4)  # class: cls_base + 4
        normalized_obj_bbox_list.append(get_object_bbox(object, image.shape))
    else:
        pass


def crete_datasets(img):
    # 获取图片路径和标签
    img_path, objects = img.get_image_data()

    # 获取图片
    _, image = img.get_image()

    # 确定图片应该放置的目标文件夹
    if random.random() < CONF.dataset_yolo.split_ratio:
        images_dataset_folder = CONF.dataset_yolo.images_train
        labels_dataset_folder = CONF.dataset_yolo.labels_train
    else:
        images_dataset_folder = CONF.dataset_yolo.images_val
        labels_dataset_folder = CONF.dataset_yolo.labels_val

    # 保存JPG图片
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    target_path = os.path.join(images_dataset_folder, img_name + '.jpg')
    cv2.imwrite(target_path, image)

    # 获取标签
    obj_cat_list = []
    normalized_obj_bbox_list = []

    for _, object in enumerate(objects):
        # 获取每个物体的特征
        attributes = object.attributes
        # 不考虑反光
        if attributes['reflection'] == 'reflected':
            continue
        else:
            # 将非正向摄像头以及unknown仅作为摄像头类
            if attributes['direction'] != 'front' or attributes['state'] == 'unknown' or attributes[
                'pictogram'] == 'unknown':
                obj_cat_list.append(0)  # class: 0
                normalized_obj_bbox_list.append(get_object_bbox(object, image.shape))
            # 基于红绿灯类型和颜色细分正向摄像头类
            elif attributes['direction'] == 'front':
                # 圆形 关/红/黄/红黄/绿
                if attributes['pictogram'] == 'circle':
                    cls_base = 1
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                # 左转 关/红/黄/红黄/绿
                elif attributes['pictogram'] == 'arrow_left':
                    cls_base = 6
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                # 右转 关/红/黄/红黄/绿
                elif attributes['pictogram'] == 'arrow_right':
                    cls_base = 11
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                # 直行 关/红/黄/红黄/绿
                elif attributes['pictogram'] == 'arrow_straight':
                    cls_base = 16
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                # 电车 关/红/黄/红黄/绿
                elif attributes['pictogram'] == 'tram':
                    cls_base = 21
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                # 行人 关/红/黄/红黄/绿
                elif attributes['pictogram'] == 'pedestrian':
                    cls_base = 26
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                # 自行车 关/红/黄/红黄/绿
                elif attributes['pictogram'] == 'bicycle':
                    cls_base = 31
                    assign_cls_and_bbox(image, object, obj_cat_list, normalized_obj_bbox_list, cls_base)
                else:
                    pass
            else:
                pass

    # txt文件的创建和写入
    target_path = os.path.join(labels_dataset_folder, img_name + '.txt')

    with open(target_path, 'w') as f:
        for obj_cat, normalized_obj_bbox_list in zip(obj_cat_list, normalized_obj_bbox_list):
            # 构建要写入的行内容
            line = (f"{obj_cat} {normalized_obj_bbox_list[0]} {normalized_obj_bbox_list[1]} "
                    f"{normalized_obj_bbox_list[2]} {normalized_obj_bbox_list[3]}\n")

            # 将行写入文件
            f.write(line)


def main(label_file, data_base_dir):
    # 读取数据集
    database = get_database(label_file, data_base_dir)

    # 创建dataset路径并确保目录存在
    os.makedirs(CONF.dataset_yolo.images_train, exist_ok=True)
    os.makedirs(CONF.dataset_yolo.images_val, exist_ok=True)
    os.makedirs(CONF.dataset_yolo.labels_train, exist_ok=True)
    os.makedirs(CONF.dataset_yolo.labels_val, exist_ok=True)

    # 获取处理器核心数量
    num_cores = multiprocessing.cpu_count()

    # 使用多进程处理数据集中的每张图片
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 创建进度条
        with tqdm(total=len(database.images), desc='Processing Images') as pbar:
            for _ in pool.imap_unordered(crete_datasets, database.images):
                pbar.update(1)

    print("所有图片处理完成")


if __name__ == "__main__":
    main(label_file=CONF.PATH.LABELS, data_base_dir=CONF.PATH.DATA)
