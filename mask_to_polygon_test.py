import os
import glob
import json
import warnings
import numpy as np
from argparse import ArgumentParser
from utils import image_to_annotation_format, NpEncoder
warnings.simplefilter('ignore', np.RankWarning)
from mask_to_polygon.mask_to_polygon import datahunt_polygon_format, MaskToPolygon

def main():
    parser = ArgumentParser()
    parser.add_argument('--sampling_ratio', default=0.1, type=float)
    parser.add_argument('--min_poly_num', default=5, type=str)
    parser.add_argument('--poly_method', default='imantics', type=str, choices=['imantics', 'contours'],
                        help='Polygon conversion method')
    parser.add_argument('--anno_path', default='data_sample/input/v3/annotation', help='Annotation file or Directory')
    parser.add_argument('--img_path', default='data_sample/input/v3/image', help='Image file or Directory')
    parser.add_argument('--output_path', default='data_sample/output', help='Output or Directory')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    mask_to_poly = MaskToPolygon(args)
    for ann_path in glob.glob(f'{args.anno_path}/json/*.json'):
        mask_ls, cls_ls, image_size = image_to_annotation_format(ann_path, f"{args.anno_path}/mask/{os.path.basename(ann_path).replace('json', 'png')}")
        polygons = mask_to_poly.convert_mask_to_polygon(mask_ls)
        output = {
            'labels': datahunt_polygon_format(cls_ls, polygons, image_size),
            'images': 'image_url',
            'relativePath': '',
            'name': os.path.basename(ann_path).replace('json', 'png')
        }
        json.dump(output, open(f"{args.output_path}/{os.path.basename(ann_path)}", 'w'), indent=4, cls=NpEncoder)


if __name__ == '__main__':
    main()