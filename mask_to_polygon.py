import os
import glob
import math
import json
import pickle
import warnings
import numpy as np
from imantics import Mask
from skimage import measure
from argparse import ArgumentParser
warnings.simplefilter('ignore', np.RankWarning)


def datahunt_polygon_format(labels, polygons, img_size):
    label_ls = []
    for label, polygon in zip(labels, polygons):
        label_ls.append({
            'label': label['class'],
            'imgSize': {'width': img_size[1], 'height': img_size[0]},
            'boundingPoly': {
                'type': 'POLYGON',
                'vertices': [{'x': x, 'y': y} for x, y in polygon]
            }
        })
    return label_ls


class MaskToPolygon:
    def __init__(self, args):
        """
        Convert segmentation masking result to polygon
        """
        self.args = args

    def create_mask(self, mask_coord, image_shape):
        """Create masking array from coordinates"""
        mask = np.zeros(image_shape)
        mask[[*mask_coord.T]] = 1
        return mask

    def imantics_poly(self, mask):
        """Convert Maks to Polygon based on imantics library"""
        poly = Mask(mask).polygons()[0]
        poly = np.array([[x, y] for x, y in zip(poly[::2], poly[1::2])])
        return poly

    def contours_poly(self, mask):
        """Convert the mask to polygon by extracting the bounds of the object"""
        binary_mask = mask.astype('uint8')
        contours = measure.find_contours(binary_mask, 0.5)
        poly = np.flip(contours[0], axis=1)
        return poly

    def calculate_slope(self, poly_ls, index):
        """Calculate the slope between two points"""
        p = np.array(poly_ls[index:index + 3])
        slope1, _ = np.polyfit(p[:2, 0], p[:2, 1], 1)
        slope2, _ = np.polyfit(p[1:3, 0], p[1:3, 1], 1)
        return np.round(slope1) == np.round(slope2)

    def sort_polygon(self, poly):
        """Sort polygon coordinates based on polar angle"""
        centroid = (np.sum(poly[:, 0]) / len(poly), np.sum(poly[:, 1]) / len(poly))
        poly = poly.tolist()
        poly.sort(key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))
        return poly

    def slope_filtering(self, poly):
        """Remove intermediate coordinates having the same slope as the anteroposterior coordinates"""
        index = 0
        while True:
            if index + 3 == len(poly):
                break
            pop_index = index + 1
            is_match = self.calculate_slope(poly, index)
            if is_match:
                poly.pop(pop_index)
            else:
                index += 1
        return poly

    def convert_mask_to_polygon(self, infer_result):
        """Convert mask to polygon"""
        polygon_result = []
        for data_info in infer_result['results']:
            if self.args.masking_format == 'bool':
                mask = data_info['mask']
            else:
                mask = self.create_mask(data_info['mask_coord'], self.infer_result['imgSize'])
            if self.args.poly_method == 'imantics':
                poly = self.imantics_poly(mask)
            else:
                poly = self.contours_poly(mask)
            unique_poly = self.sort_polygon(np.unique(poly, axis=0))
            filtered_poly = self.slope_filtering(unique_poly)
            filtering_interval = int(np.round(len(filtered_poly) / (len(filtered_poly) * self.args.sampling_ratio)))
            final_poly = filtered_poly[::filtering_interval]
            polygon_result.append(final_poly)
        return polygon_result


def main():
    parser = ArgumentParser()
    parser.add_argument('--sampling_ratio', default=0.1, type=float)
    parser.add_argument('--masking_format', default='bool', type=str, choices=['bool', 'coord'],
                        help='Segmentation model inference result format, select from bool type array or coord array')
    parser.add_argument('--poly_method', default='contours', type=str, choices=['imantics', 'contours'],
                        help='Polygon conversion method')
    parser.add_argument('--anno_path', default='data_sample/input/annotation/mask', help='Annotation file or Directory')
    parser.add_argument('--img_path', default='data_sample/input/image', help='Image file or Directory')
    parser.add_argument('--output_path', default='data_sample/output', help='Output or Directory')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    mask_to_poly = MaskToPolygon(args)
    for ext in ('jpg', 'png'):
        for img_path in glob.glob(f'{args.img_path}/*.{ext}'):
            infer_result = pickle.load(open(f"{args.anno_path}/{os.path.basename(img_path).replace(ext, 'pkl')}", 'rb'))
            polygons = mask_to_poly.convert_mask_to_polygon(infer_result)
            output = {
                'labels': datahunt_polygon_format(infer_result['results'], polygons, infer_result['imgSize']),
                'images': 'image_url',
                'relativePath': '',
                'name': os.path.basename(img_path)
            }
            json.dump(output, open(f"{args.output_path}/{os.path.basename(img_path).replace(ext, 'json')}", 'w'), indent=4)


if __name__ == '__main__':
    main()