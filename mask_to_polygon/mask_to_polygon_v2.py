import warnings
import numpy as np
from imantics import Mask
from skimage import measure
from skimage.measure import regionprops
from scipy.ndimage import label, binary_dilation
warnings.simplefilter('ignore', np.RankWarning)


def image_to_annotation_format(ann_info, mask_image):
    """Convert image to mask2polygon input format"""

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


def add_corner_points(corner_points, final_poly, poly):
    final_poly = np.array(final_poly)
    corner_points = corner_points[np.where(np.any(np.isin(corner_points, final_poly) == 0, axis=1))[0]]
    corner_idx, point_idx = [], []
    for p in corner_points:
        corner_idx.append(int(np.where((poly == p).all(axis=1))[0][0]))
    for p in final_poly:
        point_idx.append(int(np.where((poly == p).all(axis=1))[0][0]))
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
            'category': cls,
            'imgSize': {'width': img_size[1], 'height': img_size[0]},
            'boundingPoly': {
                'type': 'POLYGON',
                'vertices': [{'x': x, 'y': y} for x, y in polygon]
            }
        })
    return label_ls


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


def convert_mask_to_polygon(anno_info, mask_image, poly_method, min_poly_num, sampling_ratio):
    """Convert mask to polygon"""
    mask_ls, cls_ls, image_size = image_to_annotation_format(anno_info, mask_image)
    mask_ls, cls_ls = split_seg_result(mask_ls, cls_ls)

    polygon_result = []
    for mask in mask_ls:
        if poly_method == 'imantics':
            poly = imantics_poly(mask)
        else:
            poly = contours_poly(mask)
        corner_points = get_corner_points(poly)

        if poly_method == 'contours':
            poly = slope_filtering(poly.tolist())

        filtering_interval = int(np.round(len(poly) / (len(poly) * sampling_ratio)))
        final_poly = poly[::filtering_interval]
        final_poly = add_corner_points(corner_points, final_poly, poly)

        if len(final_poly) < min_poly_num:
            continue
        polygon_result.append(final_poly)

    output = {
        'labels': datahunt_polygon_format(cls_ls, polygon_result, image_size),
        'images': 'image_url',
        'relativePath': '',
        'name': anno_info['name']
    }

    return output

