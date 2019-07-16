import json
import os
import pandas as pd
import uuid
import cv2
from pprint import pprint

from cocoapi.PythonAPI.pycocotools.coco import COCO
import itertools

categories = set()


def parse_label(path, img_id, width, height):
    """
    将darknet label转换为coco的label
    :param path:
    :param img_id:
    :param width:
    :param height:
    :return: annotations
    """
    try:
        labels = pd.read_csv(path, delimiter=' ', header=None).values
    except Exception:
        print('Empty Label: {}'.format(path))
        return []
    annotations = []
    for idx, label in enumerate(labels):
        category_id = label[0] if isinstance(label[0], str) else str(int(label[0]))
        center_x, center_y, width_ratio, height_ratio = label[1:]
        annotations.append({
            'id': '{}_{}'.format(img_id, idx),
            'image_id': img_id,
            'category_id': category_id,
            'segmentation': None,
            'area': width * width_ratio * height * height_ratio,
            'bbox': [width * (center_x - width_ratio / 2), height * (center_y - height_ratio / 2),
                     width * width_ratio, height * height_ratio],
            'iscrowd': 0,
        })
        categories.add(str(category_id))
    return annotations



def process(dataset_type):
    assert dataset_type in ['tracking', 'object']

    dataset_root = '/media/kid/space1/{}/training'.format(dataset_type)
    img_root = os.path.join(dataset_root, 'image_02' if dataset_type == 'tracking' else 'image_2')
    label_root = os.path.join(dataset_root, 'front_view_label')

    images = []
    annotations = []
    if dataset_type == 'tracking':
        img_list = [[os.path.join(seq_root, img_path) for img_path in os.listdir(os.path.join(img_root, seq_root))]
                    for seq_root in os.listdir(img_root)]
        img_list = list(itertools.chain(*img_list))
    elif dataset_type == 'object':
        img_list = os.listdir(img_root)

    for idx, img_name in enumerate(img_list):
        img_path = os.path.join(img_root, img_name)
        height, width = cv2.imread(img_path).shape[:2]
        img_name_wo_suffix = img_name[:img_name.rfind('.')]
        label_path = os.path.join(label_root, '{}.txt'.format(img_name_wo_suffix))

        # 解析标签
        annotation = parse_label(label_path, img_id=img_name, width=width, height=height)

        images.append({
            'license': 3,
            'file_name': img_name,
            'height': height,
            'width': width,
            'id': img_name_wo_suffix,
            'coco_url': '', 'date_captured': '', 'flickr_url': '',
        })
        annotations.extend(annotation)

        #print('{}/{} {}'.format(idx, len(img_list), img_name))

    dataset = {
        'info': {
            'description': 'KITTI Object',
            'url': '',
            'version': '1.0',
            'year': 2019,
            'contributor': 'LDMC',
            'date_created': '2019/07/15'
        },
        'images': images,
        'annotations': annotations,
        "categories": [{
            "id": category_id,
            "name": str(category_id),
            "supercategory": 'str',
        } for category_id in categories]

    }
    json.dump(dataset, open('./KITTI_tracking_front_view.json', 'w'))

#dataset_type = 'object'  #
#dataset_type = 'tracking'
process('object')
process('tracking')