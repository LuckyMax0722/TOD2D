import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from lib.config import CONF


def get_obj_data(label_file):
    obj_data = []  # 存储结果的列表

    # 打开文件并按行读取
    with open(label_file, 'r') as file:
        for line in file:
            # 按空格分割行
            parts = line.split()

            # 获取类别
            category = int(parts[0])

            # 将后四位数字转换为浮点数，并放入一个列表
            bbox = [float(x) for x in parts[1:]]

            # 将类别和数字列表组成一个子列表，并添加到结果列表中
            obj_data.append(bbox)

    return obj_data


def save_plot_with_incremental_name(figure, output_folder=CONF.PATH.DEMO):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输出文件夹中现有的文件数量
    existing_files = os.listdir(output_folder)
    num_existing_files = len(existing_files)

    # 设置下一个文件的名称
    next_filename = f"demo_{num_existing_files + 1}.png"

    # 保存图形为文件
    output_path = os.path.join(output_folder, next_filename)
    figure.savefig(output_path)


def visualize(file_path, normalized_obj_bbox):
    # 读取图片
    img = mpimg.imread(file_path)

    # 创建一个新的图形
    fig, ax = plt.subplots()

    # 显示图片
    ax.imshow(img)

    # 添加坐标轴
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    for i in range(len(normalized_obj_bbox)):
        # 显示归一化的bbox
        bbox_norm = normalized_obj_bbox[i]  # 示例归一化bbox坐标
        img_height, img_width, _ = img.shape
        x_center = bbox_norm[0] * img_width
        y_center = bbox_norm[1] * img_height
        w = bbox_norm[2] * img_width
        h = bbox_norm[3] * img_height
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        rect = plt.Rectangle((x_min, y_min), w, h, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 显示
    save_plot_with_incremental_name(plt)
    plt.show()


if __name__ == '__main__':
    # 用于存储文件路径的列表
    image_file_paths = []
    label_file_paths = []

    # 遍历文件夹
    for root, dirs, files in os.walk(CONF.dataset.images_train):
        # 遍历当前文件夹下的文件
        for file_name in sorted(files):
            # 构建文件的完整路径
            file_path = os.path.join(root, file_name)
            # 将文件路径添加到列表中
            image_file_paths.append(file_path)

    for root, dirs, files in os.walk(CONF.dataset.labels_train):
        # 遍历当前文件夹下的文件
        for file_name in sorted(files):
            # 构建文件的完整路径
            file_path = os.path.join(root, file_name)
            # 将文件路径添加到列表中
            label_file_paths.append(file_path)


    '''
    for i in range(10):
        file_path = image_file_paths[i]
        label_file = label_file_paths[i]
        normalized_obj_bbox_list = get_obj_data(label_file)
    
        visualize(file_path, normalized_obj_bbox_list)
    '''
    file_path = '/home/jiachen/CARLA/dataset_yolo/dtld/images/val/DE_BBBR667_2015-04-17_14-41-09-901126_k0.jpg'
    label_file = '/home/jiachen/CARLA/dataset_yolo/dtld/labels/val/DE_BBBR667_2015-04-17_14-41-09-901126_k0.txt'
    normalized_obj_bbox_list = get_obj_data(label_file)

    visualize(file_path, normalized_obj_bbox_list)
