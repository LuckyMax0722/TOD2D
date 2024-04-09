import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from lib.config import CONF
from statistics_cls import read_data_from_txt


def flip_image(image_path, flip_img_path_list):
    # 读取图像
    image = cv2.imread(image_path)

    # 水平翻转图像
    flipped_image = cv2.flip(image, 1)

    # 创建输出目录
    output_dir = CONF.dataset_tlc_classifier.images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成新的文件名
    file_name = os.path.basename(image_path)
    flipped_file_name = "flipped_" + file_name

    # 保存翻转后的图像
    flipped_image_path = os.path.join(output_dir, flipped_file_name)
    flip_img_path_list.append(flipped_image_path)
    cv2.imwrite(flipped_image_path, flipped_image)


def augment_data(input_file):
    # 获取标签
    flip_img_path_list = []
    flip_obj_col_list = []
    flip_obj_cls_list = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Augmenting Data Flipping images", unit="images"):
        parts = line.strip().split(' ')
        image_path = parts[0]
        color = parts[1]
        direction = parts[2]

        if direction == '1':  # Left
            flip_image(image_path, flip_img_path_list)
            flip_obj_col_list.append(color)
            flip_obj_cls_list.append(2)
        elif direction == '2':  # Right
            flip_image(image_path, flip_img_path_list)
            flip_obj_col_list.append(color)
            flip_obj_cls_list.append(1)
        elif direction == '3':  # Straight
            flip_image(image_path, flip_img_path_list)
            flip_obj_col_list.append(color)
            flip_obj_cls_list.append(3)
        else:
            pass

    # 打开一个 TXT 文件进行追加写入
    with open(input_file, 'a') as f:
        # 遍历图片信息列表，并将每一行数据追加到文件末尾
        for path, col, cls in zip(flip_img_path_list, flip_obj_col_list, flip_obj_cls_list):
            line = f"{path} {col} {cls}\n"
            f.write(line)


def main(txt_file_path):
    augment_data(txt_file_path)


def plot_bar_chart(data, labels, title):
    plt.bar(labels, data)
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(os.path.join(CONF.PATH.DEMO, (title + ' After Flipping' + '.png')))
    plt.show()


def statistics(txt_file_path):
    colors_count, categories_count = read_data_from_txt(txt_file_path)

    # 绘制颜色类型统计图
    color_labels = ['Red', 'Yellow', 'Green', 'Off']
    plot_bar_chart(colors_count, color_labels, 'Color Type Statistics')

    # 绘制类别统计图
    category_labels = ['Circle', 'Left', 'Right', 'Straight', 'Other']
    plot_bar_chart(categories_count, category_labels, 'Category Statistics')


if __name__ == "__main__":
    txt_file_path = CONF.dataset_tlc_classifier.labels_txt_path

    main(txt_file_path)

    statistics(txt_file_path)
