from __future__ import print_function

import sys
from pathlib import Path

import cv2
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.dtld_parsing.driveu_dataset import DriveuDatabase
from lib.config import CONF


def visualize(file_path, objects):
    # 读取图像
    image = cv2.imread(file_path)

    # 读取标签
    for o in objects:
        cv2.rectangle(
            image,
            (o.x, o.y),
            (o.x + o.width, o.y + o.height),
            o.color_from_attributes(),
            1,
        )

    cv2.imwrite(os.path.join(CONF.PATH.DEMO, 'cv2_output_labeled.jpg'), image)

    # 显示带有边界框的图像
    cv2.imshow("Image with bounding box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(label_file, data_base_dir, mode):
    # 读取数据集
    database = DriveuDatabase(label_file)
    if not database.open(data_base_dir):
        return False

    # 读取数据集中第一张图片
    img = database.images[0]

    # 输出图片
    if not os.path.exists(CONF.PATH.DEMO):
        try:
            os.makedirs(CONF.PATH.DEMO)
        except OSError as e:
            print(f"Fail to create {CONF.PATH.DEMO} : {e}")
    else:
        print(f"Target path {CONF.PATH.DEMO} exist")

    if mode == 1:
        # 读取数据集中第一张图片的路径
        file_path, objects = img.get_image_data()
        _, image = img.get_image()
        cv2.imwrite(os.path.join(CONF.PATH.DEMO, 'cv2_output.jpg'), image)

        visualize(os.path.join(CONF.PATH.DEMO, 'cv2_output.jpg'), objects)

    elif mode == 2:
        image_with_label = img.get_labeled_image()
        cv2.imwrite(os.path.join(CONF.PATH.DEMO, 'driveu_output_labeled.jpg'), image_with_label)

        # 显示带有边界框的图像
        cv2.imshow("Image with bounding box", image_with_label)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    label_file = '/home/jiachen/CARLA/data/DTLD_Labels_v2.0/v2.0/Berlin.json'
    data_base_dir = '/home/jiachen/CARLA/data'

    # 1: use cv2 from image_path + labels
    # 2: use function from driveu
    mode = 1

    main(label_file, data_base_dir, mode)

    # press 0 to kill the image window!!
