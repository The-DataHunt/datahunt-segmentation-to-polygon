import json
import warnings
import numpy as np
from pprint import pprint
from utils import vis_polygons
import matplotlib.pyplot as plt
from mask_to_polygon.mask_to_polygon import MaskPolygonConverter
warnings.simplefilter('ignore', np.RankWarning)


def main():
    converter = MaskPolygonConverter(poly_method='imantics',
                                     min_poly_num=3,
                                     sampling_ratio=0.1)

    # auto_labeling_test
    ann_path = './data_sample/input/v4/annotation/semantic_seg_0.json'
    anno_info = json.load(open(ann_path, 'r'))
    poly_result = converter.get_auto_labeling_result(anno_info)
    pprint(poly_result)

    # polygon_to_mask_image test
    platte_path = './color_platte.json'
    color_platte = json.load(open(platte_path, 'r'))
    color_map_image = converter.polygon_to_mask_image(poly_result, anno_info['imgSize'], color_platte)
    plt.imshow(color_map_image)
    plt.show()

    # visualize polygon result
    polygons = []
    for o in poly_result:
        polygons.append([[coord['x'], coord['y']] for coord in o['boundingPoly']['vertices']])

    vis_polygons(polygons=polygons, image_path=ann_path.replace('annotation', 'image').replace('.json', '.png'))


if __name__ == '__main__':
    main()