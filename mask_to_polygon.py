import os
import glob
import json
import warnings
import numpy as np
from imantics import Mask
from skimage import measure
from argparse import ArgumentParser
from utils import image_to_annotation_format, NpEncoder
warnings.simplefilter('ignore', np.RankWarning)


def add_corner_points(corner_points, final_poly, poly):
    corner_points = corner_points[np.where(np.any(np.isin(corner_points, final_poly) == 0, axis=1))[0]]
    corner_idx, point_idx = [], []
    for p in corner_points:
        corner_idx.append(int(np.where((poly == p).all(axis=1))[0]))
    for p in final_poly:
        point_idx.append(int(np.where((poly == p).all(axis=1))[0]))
    merge_ls = sorted(point_idx + corner_idx)
    final_poly = final_poly.tolist()
    sort_idx = np.argsort(corner_idx)
    corner_idx = np.array(corner_idx)[sort_idx]
    corner_points = np.array(corner_points)[sort_idx]
    for idx, p in zip(corner_idx, corner_points):
        final_poly.insert(merge_ls.index(idx), list(p))
    return np.array(final_poly)


def get_corner_points(poly):
    rect = np.zeros((4, 2), dtype="float32")

    s = poly.sum(axis=1)
    rect[0] = poly[np.argmin(s)]
    rect[2] = poly[np.argmax(s)]

    diff = np.diff(poly, axis=1)
    rect[1] = poly[np.argmin(diff)]
    rect[3] = poly[np.argmax(diff)]
    return np.unique(rect, axis=0)


def datahunt_polygon_format(classes, polygons, img_size):
    label_ls = []
    for cls, polygon in zip(classes, polygons):
        label_ls.append({
            'label': cls,
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
        for mask in infer_result:
            if self.args.poly_method == 'imantics':
                poly = self.imantics_poly(mask)
            else:
                poly = self.contours_poly(mask)
            corner_points = get_corner_points(poly).astype('int32')

            if self.args.poly_method == 'contours':
                poly = self.slope_filtering(poly.tolist())

            filtering_interval = int(np.round(len(poly) / (len(poly) * self.args.sampling_ratio)))
            final_poly = poly[::filtering_interval]
            final_poly = add_corner_points(corner_points, final_poly, poly)

            if len(final_poly) < self.args.min_poly_num:
                continue
            polygon_result.append(final_poly)
        return polygon_result


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