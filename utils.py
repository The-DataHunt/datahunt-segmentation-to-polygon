import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def vis_polygons(polygons, image_path):
    image = cv2.imread(image_path)
    plt.imshow(image)
    for poly in polygons:
        shp = patches.Polygon(poly, fill=True, edgecolor='r', ls='solid', lw=0.5)
        plt.gca().add_patch(shp)
    plt.show()


def image_to_annotation_format(ann_path, mask_path):
    """Convert image to mask2polygon input format"""

    mask_image = cv2.imread(mask_path)
    ann_info = json.load(open(ann_path, 'r'))

    mask_ls, class_ls = [], []
    for instance in ann_info['results']:
        mask = np.zeros(mask_image.shape[:2], dtype=bool)
        color = instance['color']
        X, Y, _ = np.where((mask_image == color))
        mask[X, Y] = True
        if np.sum(mask) == 0:
            continue
        mask_ls.append(mask)
        class_ls.append(instance['class'])
    return mask_ls, class_ls, ann_info['imgSize']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    import glob

    ann_path = 'data_sample/input/v2/annotation/json'
    mask_path = 'data_sample/input/v2/annotation/mask'

    for ann_file in glob.glob(f"{ann_path}/*.json"):
        mask_ls, cls_ls, image_size = image_to_annotation_format(ann_file, f"{mask_path}/{os.path.basename(ann_file).replace('json', 'png')}")
        print(f"Mask: {len(mask_ls)}, Class: {len(cls_ls)}, ImageSize: {image_size}")