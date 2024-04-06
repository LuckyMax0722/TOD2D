import matplotlib.pyplot as plt

import os
from lib.config import CONF

class_name = {
    0: 'traffic light back',
    1: 'traffic light left',
    2: 'traffic light right',
    3: 'traffic light',
    4: 'circle.red',
    5: 'circle.yellow',
    6: 'circle.red_yellow',
    7: 'circle.green',
    8: 'arrow_left.red',
    9: 'arrow_left.yellow',
    10: 'arrow_left.red_yellow',
    11: 'arrow_left.green',
    12: 'arrow_right.red',
    13: 'arrow_right.yellow',
    14: 'arrow_right.red_yellow',
    15: 'arrow_right.green',
    16: 'arrow_straight.red',
    17: 'arrow_straight.yellow',
    18: 'arrow_straight.red_yellow',
    19: 'arrow_straight.green',
    20: 'tram.red',
    21: 'tram.yellow',
    22: 'tram.red_yellow',
    23: 'tram.green',
    24: 'pedestrian.red',
    25: 'pedestrian.yellow',
    26: 'pedestrian.red_yellow',
    27: 'pedestrian.green',
    28: 'bicycle.red',
    29: 'bicycle.yellow',
    30: 'bicycle.red_yellow',
    31: 'bicycle.green'
}


def get_txt(txt_path):
    class_count = {class_name[i]: 0 for i in range(len(class_name))}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _, label = line.strip().split()
            class_count[class_name[int(label)]] += 1

    return class_count


if __name__ == '__main__':
    labels_count = get_txt(CONF.dataset_tlc_classifier.labels_txt_path)
    classes = list(labels_count.keys())
    counts = list(labels_count.values())

    plt.figure(figsize=(10, 8))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Classes')
    plt.xticks(rotation=90)

    # 调整底部留白
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(os.path.join(CONF.PATH.DEMO, 'visualisation_cls.png'))
    plt.show()
