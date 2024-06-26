from __future__ import print_function

import sys
from pathlib import Path

import cv2
import os

import multiprocessing
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.dtld_parsing.driveu_dataset import DriveuDatabase
from lib.config import CONF


def get_database(label_file, data_base_dir):
    # 读取数据集
    database = DriveuDatabase(label_file)
    if not database.open(data_base_dir):
        return False

    return database


def xywh2xyxy(x):
    # Convert nx4 boxes from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = [0, 0, 0, 0]

    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2]  # bottom right x
    y[3] = x[1] + x[3]  # bottom right y

    return y


def get_resize_object_bbox(object):
    # Reshape and pad cutouts
    b = [object.x, object.y, object.width, object.height]  # boxes

    # rectangle to square
    b_length = max(b[2], b[3])
    # pad
    b[2] = b_length + 7
    b[3] = b_length + 7

    b = xywh2xyxy(b)
    b = [int(x) for x in b]

    return b


def get_bbox_image(img, img_path, object, object_idx):
    resize_bbox = get_resize_object_bbox(object)

    # 判断bbox是否能够截取到合理的图像
    if resize_bbox[0] < 0 or resize_bbox[1] < 0 or resize_bbox[2] > img.shape[1] or resize_bbox[3] > img.shape[0]:
        return False, ''
    else:
        cutout = img[resize_bbox[1]:resize_bbox[3], resize_bbox[0]:resize_bbox[2]]
        im = cv2.resize(cutout, (224, 224))  # BGR

        # 保存JPG图片
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        target_path = os.path.join(CONF.dataset_tlc_classifier.images, img_name + '_' + str(object_idx) + '.jpg')
        cv2.imwrite(target_path, im)

        # visualization
        '''
        cv2.imshow('Image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        return True, target_path


def assign_cls(object):
    attributes = object.attributes
    if attributes['state'] == 'red':
        return 0
    elif attributes['state'] == 'yellow':
        return 1
    elif attributes['state'] == 'green':
        return 2
    else:
        raise ValueError('Unknown state')


def crete_datasets(img):
    # 获取图片路径和标签
    image_path, objects = img.get_image_data()

    # 获取图片
    _, image = img.get_image()

    # 获取标签
    img_path_list = []
    obj_col_list = []
    obj_cls_list = []


    for idx, object in enumerate(objects):
        # 获取每个物体的特征
        attributes = object.attributes

        # 不考虑反光
        if attributes['reflection'] == 'reflected':
            continue
        elif attributes['state'] == 'red_yellow':
            pass
        else:
            # 将非正向摄像头作为off / other
            if attributes['direction'] != 'front' or attributes['state'] == 'unknown' or attributes['state'] == 'off':
                state, target_path = get_bbox_image(image, image_path, object, idx)
                if state:
                    obj_col_list.append(3)  # off
                    obj_cls_list.append(4)  # other
                    img_path_list.append(target_path)
                else:
                    continue

            elif attributes['pictogram'] == 'circle':
                # 圆形 红/黄/绿
                state, target_path = get_bbox_image(image, image_path, object, idx)
                if state:
                    obj_col_list.append(assign_cls(object))
                    obj_cls_list.append(0)  # Circle
                    img_path_list.append(target_path)
                else:
                    continue

            elif attributes['pictogram'] == 'arrow_left':
                # 左 红/黄/绿
                state, target_path = get_bbox_image(image, image_path, object, idx)
                if state:
                    obj_col_list.append(assign_cls(object))
                    obj_cls_list.append(1)  # Left
                    img_path_list.append(target_path)
                else:
                    continue

            elif attributes['pictogram'] == 'arrow_right':
                # 右 红/黄/绿
                state, target_path = get_bbox_image(image, image_path, object, idx)
                if state:
                    obj_col_list.append(assign_cls(object))
                    obj_cls_list.append(2)  # Right
                    img_path_list.append(target_path)
                else:
                    continue

            elif attributes['pictogram'] == 'arrow_straight':
                # 直行 红/黄/绿
                state, target_path = get_bbox_image(image, image_path, object, idx)
                if state:
                    obj_col_list.append(assign_cls(object))
                    obj_cls_list.append(3)  # Straight
                    img_path_list.append(target_path)
                else:
                    continue

            elif attributes['pictogram'] == 'unknown' or attributes['pictogram'] == 'tram' or attributes['pictogram'] == 'pedestrian' or attributes['pictogram'] == 'bicycle':
                # 其他 红/黄/绿
                state, target_path = get_bbox_image(image, image_path, object, idx)
                if state:
                    obj_col_list.append(assign_cls(object))
                    obj_cls_list.append(4)  # Other
                    img_path_list.append(target_path)
                else:
                    continue
            else:
                pass

    # txt文件的创建
    txt_output_path = CONF.dataset_tlc_classifier.labels_txt_path

    # 打开一个 TXT 文件进行追加写入
    with open(txt_output_path, 'a') as f:
        # 遍历图片信息列表，并将每一行数据追加到文件末尾
        for path, col, cls in zip(img_path_list, obj_col_list, obj_cls_list):
            line = f"{path} {col} {cls}\n"
            f.write(line)


def main(label_file, data_base_dir):
    # 读取数据集
    database = get_database(label_file, data_base_dir)

    # 创建dataset路径并确保目录存在
    os.makedirs(CONF.dataset_tlc_classifier.images, exist_ok=True)
    os.makedirs(CONF.dataset_tlc_classifier.labels, exist_ok=True)

    # 获取处理器核心数量
    num_cores = multiprocessing.cpu_count()

    # 使用多进程处理数据集中的每张图片
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 tqdm 创建进度条
        with tqdm(total=len(database.images), desc='Processing DTLD Images') as pbar:
            for _ in pool.imap_unordered(crete_datasets, database.images):
                pbar.update(1)

    print("Finish Processing DTLD Images")


if __name__ == "__main__":
    main(label_file=CONF.PATH.LABELS_DTLD, data_base_dir=CONF.PATH.DATA_DTLD)
