import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import regionprops
from scipy.ndimage import label, binary_dilation


def split_seg_result(mask_ls, class_ls, area_threshold=4000):
    new_mask_ls, new_class_ls = [], []
    for cls, mask in zip(class_ls, mask_ls):
        grow = binary_dilation(mask, structure=np.ones((5, 5), dtype=int))
        lbl, npatches = label(grow)
        lbl[mask == 0] = 0
        for region in regionprops(lbl):
            if region.area > area_threshold:
                new_mask_ls.append(lbl == region.label)
                new_class_ls.append(cls)
    return new_mask_ls, new_class_ls


def vis_polygons(polygons, image_path, save_path=None):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    for poly in polygons:
        shp = patches.Polygon(poly, fill=False, edgecolor='#0000FF', ls='solid', lw=3, alpha=0.5)  # facecolor=color
        plt.gca().add_patch(shp)
        # plt.scatter(poly[:, 0], poly[:, 1], s=5)
    if save_path is None:
        plt.show()
    else:
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.clf()


def image_to_annotation_format(ann_path, mask_path):
    """Convert image to mask2polygon input format"""

    mask_image = cv2.imread(mask_path)
    ann_info = json.load(open(ann_path, 'r'))

    mask_ls, class_ls = [], []
    for instance in ann_info['results']:
        mask = np.zeros(mask_image.shape[:2], dtype=bool)
        color = np.array([instance['color'][key] for key in ['r', 'g', 'b']])
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