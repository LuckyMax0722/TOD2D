import os.path

import matplotlib.pyplot as plt
import numpy as np
from lib.config import CONF


def read_data_from_txt(txt_file):
    colors_count = [0, 0, 0, 0]  # 列表用于统计不同颜色类型的数量
    categories_count = [0, 0, 0, 0, 0]  # 列表用于统计不同类别的数量

    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 分割每行数据
            color = int(parts[1])  # 提取颜色类型
            category = int(parts[2])  # 提取类别

            colors_count[color] += 1  # 更新颜色类型统计
            categories_count[category] += 1  # 更新类别统计

    return colors_count, categories_count


def plot_bar_chart(data, labels, title):
    plt.bar(labels, data)
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(os.path.join(CONF.PATH.DEMO, (title + '.png')))
    plt.show()


if __name__ == "__main__":
    txt_file_path = CONF.dataset_tlc_classifier.labels_txt_path

    colors_count, categories_count = read_data_from_txt(txt_file_path)

    # 绘制颜色类型统计图
    color_labels = ['Red', 'Yellow', 'Green', 'Off']
    plot_bar_chart(colors_count, color_labels, 'Color Type Statistics')

    # 绘制类别统计图
    category_labels = ['Circle', 'Left', 'Right', 'Straight', 'Other']
    plot_bar_chart(categories_count, category_labels, 'Category Statistics')
