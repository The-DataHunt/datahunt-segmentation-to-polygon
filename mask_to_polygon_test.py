import cv2
import json
import warnings
import numpy as np
from mask_to_polygon.mask_to_polygon import convert_mask_to_polygon as v1
from mask_to_polygon.mask_to_polygon_v2 import convert_mask_to_polygon as v2
warnings.simplefilter('ignore', np.RankWarning)


def main():
    anno_info1 = json.load(open('./data_sample/input/v3/annotation/json/semantic_seg_0.json', 'r'))
    mask_image1 = cv2.imread('./data_sample/input/v3/annotation/mask/semantic_seg_0.png')
    output = v1(anno_info=anno_info1,
                mask_image=mask_image1,
                poly_method='imantics',
                min_poly_num=5,
                sampling_ratio=0.2)
    print(output)

    anno_info2 = json.load(open('./data_sample/input/re3imagesegmentation/annotation/json/sample.json', 'r'))
    mask_image2 = cv2.imread('./data_sample/input/re3imagesegmentation/annotation/mask/sample.png')
    output2 = v2(anno_info=anno_info2,
                 mask_image=mask_image2,
                 poly_method='contours',
                 min_poly_num=5,
                 sampling_ratio=0.2)
    print(output2)


if __name__ == '__main__':
    main()