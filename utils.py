import numpy as np
import random
import os


def read_image_label_txt(image_dir, txt_dir):
    txt_file = os.path.join(txt_dir, 'train.txt')
    image_paths, image_labels = [], []
    with open(txt_file) as fr:
        lines = fr.readlines()
        for line in lines:
            line.strip()
            item = line.split()
            image_paths.append(os.path.join(image_dir, item[0]))
            image_labels.append(np.array(item[1], dtype=np.int))

    return image_paths, image_labels


def read_image_label_pair_txt(image_dir, txt_dir):
    label_pair_file = os.path.join(txt_dir, 'train_label_pair.txt')
    with open(label_pair_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)
    label_pairs = []
    for line in lines:
        label_pair = []
        items = line.split()
        label_pair.append(int(items[0]))
        label_pair.append(int(items[1]))
        label_pairs.append(label_pair)

    group_lists = [
        os.path.join(txt_dir, 'train_age_group_0.txt'),
        os.path.join(txt_dir, 'train_age_group_1.txt'),
        os.path.join(txt_dir, 'train_age_group_2.txt'),
        os.path.join(txt_dir, 'train_age_group_3.txt'),
        os.path.join(txt_dir, 'train_age_group_4.txt')
    ]

    label_group_images = []
    for i in range(len(group_lists)):
        with open(group_lists[i], 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        group_images = []
        for l in lines:
            items = l.split()
            group_images.append(os.path.join(image_dir, items[0]))
        label_group_images.append(group_images)

    image_pairs = []
    for pair in label_pairs:
        image_pair = []
        image_pair.append(random.sample(label_group_images[pair[0]], 1)[0])
        image_pair.append(random.sample(label_group_images[pair[1]], 1)[0])
        image_pairs.append(image_pair)

    return label_pairs, image_pairs
