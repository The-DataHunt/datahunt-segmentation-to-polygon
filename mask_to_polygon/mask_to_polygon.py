import cv2
import base64
import warnings
import numpy as np
from imantics import Mask
import numpy_indexed as npi
from typing import List, Dict, Union
from skimage import measure, draw
warnings.simplefilter('ignore', np.RankWarning)

"""Instance Segmentation to Polygon Conversion"""


def add_corner_points(corner_points, corner_idx, final_poly, poly, poly_backup=None):
    assert len(corner_points) == len(corner_idx)
    not_exist_idx = np.where(np.any(np.isin(corner_points, final_poly) == 0, axis=1))[0]
    not_exist_corner_idx = corner_idx[not_exist_idx]
    not_exist_corner_points = corner_points[not_exist_idx]

    if poly_backup is not None:
        final_poly_point_idx = npi.indices(poly_backup, final_poly)
    else:
        final_poly_point_idx = npi.indices(poly, final_poly)

    merge_sort_idx = np.argsort(np.append(final_poly_point_idx, not_exist_corner_idx))
    merge_value = np.append(final_poly, not_exist_corner_points, axis=0)[merge_sort_idx]
    assert len(merge_sort_idx) == len(merge_value)
    return merge_value


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
            poly = np.delete(poly, pop_index, axis=0)
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
        contours, hierarchy = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

    def mask_array_to_polygon(self, mask_array: np.ndarray, sampling_ratio: float = 0.1, model_infer_object: bool = False) -> np.ndarray:
        poly_backup = None
        if self.poly_method == 'imantics':
            poly = imantics_poly(mask_array)
        else:
            poly = contours_poly(mask_array).astype('int32')

        corner_points = cv2.convexHull(poly, clockwise=True).reshape(-1, 2)
        corner_idx = npi.indices(poly, corner_points)

        if self.poly_method == 'contours':
            poly_backup = poly.copy()
            poly = slope_filtering(poly)

        filtering_interval = int(np.round(len(poly) / (len(poly) * sampling_ratio)))
        final_poly = poly[::filtering_interval]
        final_poly = add_corner_points(corner_points, corner_idx, final_poly, poly, poly_backup)

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