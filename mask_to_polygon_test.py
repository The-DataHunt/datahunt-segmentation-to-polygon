import json
import warnings
import numpy as np
from pprint import pprint
from utils import vis_polygons
from mask_to_polygon.mask_to_polygon import convert_mask_to_polygon
warnings.simplefilter('ignore', np.RankWarning)


def main():
    ann_path = './data_sample/input/v4/annotation/semantic_seg_0.json'
    anno_info = json.load(open(ann_path, 'r'))
    output = convert_mask_to_polygon(anno_info=anno_info,
                                     poly_method='imantics',
                                     min_poly_num=5,
                                     sampling_ratio=0.2)
    pprint(output)

    # test
    polygons = []
    for o in output:
        polygons.append([[coord['x'], coord['y']] for coord in o['boundingPoly']['vertices']])

    vis_polygons(polygons=polygons, image_path=ann_path.replace('annotation', 'image').replace('.json', '.png'))


if __name__ == '__main__':
    main()