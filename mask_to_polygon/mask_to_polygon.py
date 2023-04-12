import cv2
import base64
import warnings
import numpy as np
from imantics import Mask
from typing import List, Dict, Union
from skimage import measure, draw
warnings.simplefilter('ignore', np.RankWarning)

"""Instance Segmentation to Polygon Conversion"""

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


def imantics_poly(mask):
    """Convert Maks to Polygon based on imantics library"""
    poly = Mask(mask).polygons()[0]
    poly = np.array([[x, y] for x, y in zip(poly[::2], poly[1::2])])
    return poly


def contours_poly(mask):
    """Convert the mask to polygon by extracting the bounds of the object"""
    binary_mask = mask.astype('uint8')
    contours = measure.find_contours(binary_mask, 0.5)
    poly = np.flip(contours[0], axis=1)
    return poly


def calculate_slope(poly_ls, index):
    """Calculate the slope between two points"""
    p = np.array(poly_ls[index:index + 3])
    slope1, _ = np.polyfit(p[:2, 0], p[:2, 1], 1)
    slope2, _ = np.polyfit(p[1:3, 0], p[1:3, 1], 1)
    return np.round(slope1) == np.round(slope2)


def slope_filtering(poly):
    """Remove intermediate coordinates having the same slope as the anteroposterior coordinates"""
    index = 0
    while True:
        if index + 3 == len(poly):
            break
        pop_index = index + 1
        is_match = calculate_slope(poly, index)
        if is_match:
            poly.pop(pop_index)
        else:
            index += 1
    return poly


def datahunt_polygon_format(classes, polygons, img_size):
    label_ls = []
    for cls, polygon in zip(classes, polygons):
        label_ls.append({
            'label': cls,
            'imgSize': img_size,
            'boundingPoly': {
                'type': 'POLYGON',
                'vertices': [{'x': x, 'y': y} for x, y in polygon]
            }
        })
    return label_ls


class MaskPolygonConverter:
    def __init__(self, poly_method: str = 'imantics', min_poly_num: int = 3):
        """
        Admin parameters
        :param poly_method: imantics or contours
        :param min_poly_num: minimum number of points a polygon
        """
        self.poly_method = poly_method
        self.min_poly_num = min_poly_num

    def string_image_to_array(self, str_image: str, return_mask_array: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        from_buffer = np.frombuffer(base64.b64decode(str_image), np.uint8)
        mask_image = cv2.imdecode(from_buffer, cv2.IMREAD_COLOR)

        if return_mask_array:
            mask_array = np.zeros(mask_image.shape[:2], dtype=bool)
            X, Y, _ = np.where((mask_image != [0, 0, 0]))
            mask_array[X, Y] = True
            return mask_array
        return mask_image

    def check_is_multiple_or_hierarchy(self, mask_array: np.ndarray) -> List[Dict]:
        contours, hierarchy = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        parent_idx = np.squeeze(np.where(hierarchy[0, :, 3] == -1))
        mask_array_dict_ls = []
        for p_idx in np.nditer(parent_idx):
            contour_dict = dict()
            contour_dict.update({'parent': draw.polygon2mask(mask_array.shape, contours[p_idx].reshape(-1, 2)[:, [1, 0]])})
            child_idx = np.squeeze(np.where(hierarchy[0, :, 3] == p_idx))
            if child_idx.size > 0:
                contour_dict.update({'child': [draw.polygon2mask(mask_array.shape, contours[c_idx].reshape(-1, 2)[:, [1, 0]]) for c_idx in np.nditer(child_idx)]})
            mask_array_dict_ls.append(contour_dict)
        return mask_array_dict_ls

    def mask_array_to_polygon(self, mask_array: np.ndarray, sampling_ratio: float, model_infer_object: bool = False) -> np.ndarray:
        if self.poly_method == 'imantics':
            poly = imantics_poly(mask_array)
        else:
            poly = contours_poly(mask_array)

        corner_points = get_corner_points(poly).astype('int32')

        if self.poly_method == 'contours':
            poly = slope_filtering(poly.tolist())

        filtering_interval = int(np.round(len(poly) / (len(poly) * sampling_ratio)))
        final_poly = poly[::filtering_interval]
        final_poly = add_corner_points(corner_points, final_poly, poly)

        if len(final_poly) < self.min_poly_num:
            return np.empty(shape=(0, 0))
        return final_poly if model_infer_object else final_poly.flatten()

    def polygon_to_mask_image(self, polygon: List[Dict], img_size: Dict, color_platte: Dict) -> np.ndarray:
        color_map_image = np.zeros((img_size['height'], img_size['width'], 3), dtype=np.uint8)
        for info in polygon:
            label = info['label']
            poly = np.array([[v['y'], v['x']] for v in info['boundingPoly']['vertices']], dtype=np.int32)
            mask = draw.polygon2mask((img_size['height'], img_size['width']), poly)
            color_map_image[mask == 1, :] = color_platte[label]
        color_map_image = color_map_image[..., ::-1]
        return color_map_image

    def get_auto_labeling_result(self, model_inference_result_json: Dict, sampling_ratio: float = 0.1) -> List:
        mask_image = self.string_image_to_array(model_inference_result_json['mask']['value'], return_mask_array=False)

        mask_ls, class_ls = [], []
        for instance in model_inference_result_json['objects']:
            mask = np.zeros(mask_image.shape[:2], dtype=bool)
            color = np.array([instance['color'][key] for key in ['r', 'g', 'b']])
            X, Y, _ = np.where((mask_image == color))
            mask[X, Y] = True
            if np.sum(mask) == 0:
                continue
            mask_ls.append(mask)
            class_ls.append(instance['label'])

        polygon_result = []
        for mask in mask_ls:
            polygon_result.append(self.mask_array_to_polygon(mask, sampling_ratio, model_infer_object=True))
        return datahunt_polygon_format(class_ls, polygon_result,  model_inference_result_json['imgSize'])

    def get_single_object_polygon_result(self, str_image: str, sampling_ratio: float = 0.1) -> List[Dict]:
        mask_array = self.string_image_to_array(str_image, return_mask_array=True)
        mask_array_dict_ls = self.check_is_multiple_or_hierarchy(mask_array)
        for p_idx, mask_dict in enumerate(mask_array_dict_ls):
            mask_array_dict_ls[p_idx]['parent'] = self.mask_array_to_polygon(mask_dict['parent'], sampling_ratio)
            if 'child' in mask_dict.keys():
                for c_idx, child_mask in enumerate(mask_dict['child']):
                    mask_array_dict_ls[p_idx]['child'][c_idx] = self.mask_array_to_polygon(child_mask, sampling_ratio)
        return mask_array_dict_ls