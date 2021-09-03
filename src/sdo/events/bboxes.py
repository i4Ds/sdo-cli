import csv
import matplotlib.patches as mpatches
import datetime as dt
import os
from shapely.ops import unary_union
import cv2
import numpy as np
from shapely.geometry.polygon import Polygon
from datetime import datetime
from shapely import wkt
from shapely.geometry.point import Point
from shapely.geometry import Polygon
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
from sdo.events.event_loader import HEKEventManager
import logging

logger = logging.getLogger('HEKEventAnalyzer')


class EVENT_TYPE:
    """
    The event types of the solar events. (More: https://www.lmsal.com/hek/VOEvent_Spec.html)
    """
    AR = 'ar'  # Active Region
    CH = 'ch'  # Coronal Hole
    FI = 'fi'  # Filament, Kanzelhöhe

    CE = 'ce'  # Coronal Mass Ejection (CME)
    FL = 'fl'  # Flare
    SG = "sg"  # Sigmoid

    @staticmethod
    def convert(et):
        return {
            'ar': EVENT_TYPE.AR,
            'ch': EVENT_TYPE.CH,
            'ce': EVENT_TYPE.CE,
            'fi': EVENT_TYPE.FI,
            'fl': EVENT_TYPE.FL,
        }.get(et, 'ar')  # default is 'ar'


class SpatioTemporalEvent:
    def __init__(self,
                 event_type: EVENT_TYPE,
                 start_time: datetime,
                 end_time: datetime,
                 hpc_coord: (Point, str),
                 hpc_bbox: (Polygon, str),
                 hpc_boundcc: (Polygon, str),
                 kb_archivid: str):
        """
        :param event_type: The event for which the results are returned.
        :param start_time: Start time of the event type.
        :param end_time: End Time of the event Type.
        :param hpc_coord: coordinates of the center of the bounding box.
        :param hpc_bbox: bounding box of the polygon.
        :param hpc_boundcc: polygon of the detected event (if present).
        :param kb_archivid: Unique id for each event type.
        """
        self.event_type: EVENT_TYPE = event_type
        self.start_time: datetime = start_time
        self.end_time: datetime = end_time
        self.hpc_coord: (Point, str) = hpc_coord
        self.hpc_bbox: (Polygon, str) = hpc_bbox
        if(hpc_boundcc):
            self.hpc_boundcc: (Polygon, str) = hpc_boundcc
        self.kb_archivid: str = kb_archivid

    @classmethod
    def from_dict(cls, instance: dict):
        obj = cls.__new__(cls)
        super(SpatioTemporalEvent, obj).__init__()
        obj.event_type = instance['event_type']
        obj.start_time = instance['event_starttime']
        obj.end_time = instance['event_endtime']
        obj.hpc_coord = instance['hpc_coord']
        obj.hpc_bbox = instance['hpc_bbox']
        if instance['hpc_boundcc'] != '':
            obj.hpc_boundcc = instance['hpc_boundcc']
        else:
            obj.hpc_boundcc = None
        obj.kb_archivid = instance['kb_archivid']
        return obj

    @property
    def event_type(self):
        return self.__event_type

    @event_type.setter
    def event_type(self, event_type):
        if isinstance(event_type, EVENT_TYPE):
            self.__event_type = event_type
        elif isinstance(event_type, str):
            self.__event_type = EVENT_TYPE.convert(event_type)

    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        if isinstance(start_time, datetime):
            self.__start_time = start_time
        elif isinstance(start_time, str):
            self.__start_time = datetime.strptime(
                start_time, '%Y-%m-%dT%H:%M:%S')
        else:
            raise AttributeError

    @property
    def end_time(self):
        return self.__end_time

    @end_time.setter
    def end_time(self, end_time):
        if isinstance(end_time, datetime):
            self.__end_time = end_time
        elif isinstance(end_time, str):
            self.__end_time = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
        else:
            raise AttributeError

    @property
    def hpc_coord(self):
        return self.__hpc_coord

    @hpc_coord.setter
    def hpc_coord(self, hpc_coord):
        if isinstance(hpc_coord, Point):
            self.__hpc_coord = hpc_coord
        elif isinstance(hpc_coord, str):
            self.__hpc_coord = wkt.loads(hpc_coord)
        else:
            raise AttributeError

    @property
    def hpc_bbox(self):
        return self.__hpc_bbox

    @hpc_bbox.setter
    def hpc_bbox(self, hpc_bbox):
        if isinstance(hpc_bbox, Polygon):
            self.__hpc_bbox = hpc_bbox
        elif isinstance(hpc_bbox, str):
            self.__hpc_bbox = wkt.loads(hpc_bbox)
        else:
            raise AttributeError

    @property
    def hpc_boundcc(self):
        return self.__hpc_boundcc

    @hpc_boundcc.setter
    def hpc_boundcc(self, hpc_boundcc):
        if hpc_boundcc is None:
            self.__hpc_boundcc = None
        if isinstance(hpc_boundcc, Polygon):
            self.__hpc_boundcc = hpc_boundcc
        elif isinstance(hpc_boundcc, str):
            self.__hpc_boundcc = wkt.loads(hpc_boundcc)
        else:
            # raise AttributeError
            self.__hpc_boundcc = None

    @property
    def kb_archivid(self):
        return self.__kb_archivid

    @kb_archivid.setter
    def kb_archivid(self, kb_archivid):
        self.__kb_archivid = kb_archivid

    def to_dict(self):
        dict = {'event_type': self.event_type,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'hpc_coord': self.hpc_coord,
                'hpc_bbox': self.hpc_bbox,
                'hpc_boundcc': self.hpc_boundcc,
                'kb_archivid': self.kb_archivid
                }
        return dict


def convert_boundingpoints_to_pixelunit(polygon: Polygon, cdelt1, cdelt2, crpix1, crpix2, original_w, shrinkage_ratio=1):
    """
    This method converts the points coordinates from arc-sec unit to pixel unit, and meanwhile
    makes 2 modifications:
        1. Vertical mirror of the points (this is required if JPG format is being used)
        2. Shrinkage of points (required if downsized images are being used.)

    :param polygon: a list of points forming a closed shape.
    :param cdelt1: fits/jp2 header information to scale in x direction
    :param cdelt2: fits/jp2 header information to scale in y direction
    :param crpix1: fits/jp2 header information to shift in x direction
    :param crpix2: fits/jp2 header information to shift in y direction
    :param original_w: the width of the original images. It is assumed that the images
    are in square shape, hence width and height are equal.
    :param shrinkage_ratio: a float point that indicates the ratio (original_w/new_size).
    For example, for 512 X 512 image, it should be 4.0.
    :return: a polygon object (from Shapely package) and None if the list was empty. If you need
            a list of tuples (x, y) instead, you can convert it using `poly = poly.exterior.coords'
    """
    points = polygon.exterior.coords
    b = [(float(v[0]) / cdelt1 + crpix1, float(v[1]) / cdelt2 + crpix2)
         for v in points]

    # Shrink and then mirror vertically
    b = [(v[0] / shrinkage_ratio, (original_w - v[1]) / shrinkage_ratio)
         for v in b]

    return Polygon(b)


def convert_events_to_pixelunits(events_df: pd.DataFrame, img_header, original_width=4096, shrinkage_factor=2):
    all_polygons = []
    all_bboxes = []

    for i, event in events_df.iterrows():
        ste = SpatioTemporalEvent.from_dict(event)

        if(ste.hpc_boundcc is not None):
            # hpc_boundcc is optional
            poly_converted = convert_boundingpoints_to_pixelunit(polygon=ste.hpc_boundcc,
                                                                 cdelt1=img_header['CDELT'],
                                                                 cdelt2=img_header['CDELT'],
                                                                 crpix1=img_header['X0'],
                                                                 crpix2=img_header['Y0'],
                                                                 original_w=4096,
                                                                 shrinkage_ratio=2)
            all_polygons.append(poly_converted)

        bbox_converted = convert_boundingpoints_to_pixelunit(polygon=ste.hpc_bbox,
                                                             cdelt1=img_header['CDELT'],
                                                             cdelt2=img_header['CDELT'],
                                                             crpix1=img_header['X0'],
                                                             crpix2=img_header['Y0'],
                                                             original_w=4096,
                                                             shrinkage_ratio=2)

        all_bboxes.append(bbox_converted)

    return (all_polygons, all_bboxes)


def save_fig_with_bounding_boxes(src_img_path: Path, polygons, bboxes, aia_wave, timestamp):
    img = Image.open(src_img_path)
    img = img.convert('RGB')
    img_draw = ImageDraw.Draw(img)

    fig = plt.figure(figsize=(15, 15), facecolor='burlywood')

    for poly in polygons:
        poly = poly.exterior.coords
        img_draw.line(poly, fill="red", width=3)
        for point in poly:
            img_draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red")

    for bbox in bboxes:
        bbox = bbox.exterior.coords
        img_draw.line(bbox, fill="blue", width=3)
        for point in bbox:
            img_draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="blue")

    plt.axis('off')
    plt.title('{}A | {}'.format(aia_wave, timestamp))
    plt.imshow(img)


def get_meta_info(img_file_name, header_path):
    meta_df = pd.read_csv(header_path)
    header = meta_df[meta_df["FILE_NAME"] == img_file_name].iloc[0]
    return header


def convert_contours_to_polygons(contours, scale=1, min_width=3, min_height=3, max_width=48, max_height=48):
    polygons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w < min_width or h < min_height or w > max_width or h > max_height):
            continue

        x = x * scale
        y = y * scale
        w = w * scale
        h = h * scale

        polygon = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
        polygons.append(polygon)

    return polygons


def extract_bounding_boxes_from_anomaly_map(map_path,
                                            mode="simple",
                                            mask_threshold=20,
                                            scale=1,
                                            min_width=2,
                                            min_height=2,
                                            max_width=48,
                                            max_height=48,
                                            gaussian_filter=False,
                                            gaussian_filter_size=(5, 5)):
    """
    scale: the scale factor for bounding boxes (default 1)
    """

    pred_img = Image.open(map_path).convert("L")
    pred_img_arr = np.asarray(pred_img)

    if gaussian_filter:
        pred_img_arr = cv2.GaussianBlur(pred_img_arr, gaussian_filter_size, 0)

    if mode == "simple":
        mask_img = pred_img_arr > mask_threshold
        inverted_pred_img_arr = np.zeros_like(pred_img_arr)
        inverted_pred_img_arr[mask_img] = pred_img_arr[mask_img]
        mask_img = inverted_pred_img_arr
    elif mode == "binary":
        ret, mask_img = cv2.threshold(
            pred_img_arr, mask_threshold, 255, cv2.THRESH_BINARY)
    elif mode == "otsu":
        ret, mask_img = cv2.threshold(
            pred_img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        raise f"thresholding method {mode} not recognized"

    contours, hierarchy = cv2.findContours(
        mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    model_polygons = convert_contours_to_polygons(
        contours, scale=scale, min_width=min_width, min_height=min_height, max_width=max_width, max_height=max_height)

    return model_polygons


def calculate_iou(hek_bboxes, anomaly_bboxes):
    ground_truth_union = unary_union(hek_bboxes)
    predicted_union = unary_union(anomaly_bboxes)
    resulting_intersection = ground_truth_union.intersection(predicted_union)

    iou = resulting_intersection.area / \
        ground_truth_union.union(predicted_union).area

    return iou


def save_fig_with_hek_bounding_boxes_and_anomalies(img_path, hek_bboxes, hek_polygons, anomaly_bboxes, out_dir, aia_wave=171):
    plt.clf()
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_draw = ImageDraw.Draw(img)

    # facecolor='burlywood'
    fig = plt.figure(figsize=(12, 12))

    for poly in hek_polygons:
        poly = poly.exterior.coords
        img_draw.line(poly, fill="red", width=3)
        for point in poly:
            img_draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red")

    for bbox in hek_bboxes:
        bbox = bbox.exterior.coords
        img_draw.line(bbox, fill="blue", width=3)
        for point in bbox:
            img_draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="blue")

    for poly in anomaly_bboxes:
        poly = poly.exterior.coords
        img_draw.line(poly, fill="green", width=3)

    plt.axis('off')
    title = img_path.name
    plt.title(title)
    fig_path = out_dir / Path(title)

    red_patch = mpatches.Patch(color='red', label='HEK bounding box')
    blue_patch = mpatches.Patch(color='blue', label='HEK exact bounding box')
    green_patch = mpatches.Patch(
        color='green', label='Model out of distribution detection')
    plt.legend(handles=[red_patch, blue_patch, green_patch])

    plt.imshow(img)
    plt.savefig(fig_path)
    plt.close(fig)


date_format = '%Y-%m-%dT%H%M%S'


def compute_ious(src_img_path: Path,
                 sood_map_path: Path,
                 out_dir: Path,
                 db_connection_string: str,
                 aia_wave=171,
                 hek_event_types=['AR'],
                 save_fig=True):
    # result should be a csv detailing all intersection over unions
    # all calculations are done based on images that are 2048 x 2048
    if os.path.exists(src_img_path) and not os.path.isdir(src_img_path):
        raise ValueError(src_img_path + " is not a directory")

    current_date = dt.datetime.now().strftime(date_format)
    out_dir = out_dir / Path(current_date)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info(f"writing outouts to {out_dir}")

    iou_path = out_dir / Path("iou.csv")
    result_cols = ['image', 'iou', 'n_hek_events', 'n_predicted_events']
    with open(iou_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=result_cols)
        writer.writeheader()

    src_images = list(Path(src_img_path).rglob(f'*__{aia_wave}.jpeg'))

    ious = []
    for src_img in src_images:
        try:
            header = get_meta_info(
                src_img.name, src_img_path / Path("meta.csv"))
            loader = HEKEventManager(db_connection_string)
            timestamp_str = src_img.name.split("__")[0]
            timestamp = dt.datetime.strptime(timestamp_str, date_format)
            events_df = loader.find_events_at(
                timestamp, observatory="SDO", instrument="AIA", event_types=hek_event_types)

            if len(events_df) < 1:
                logger.warn(
                    f"no events found")
                continue

            # filter events that were observed in the respective wavelength, possibly also filter by feature extraction method
            events_df = events_df[events_df['obs_channelid'].str.contains(
                "171")]
            logger.info(
                f"after filter {len(events_df)} events")

            hek_bboxes, hek_polygons = convert_events_to_pixelunits(
                events_df, header)
            map_path = sood_map_path / Path(src_img.name)
            anomaly_boxes = extract_bounding_boxes_from_anomaly_map(
                map_path, mode="otsu", scale=8, gaussian_filter=False)

            if save_fig:
                save_fig_with_hek_bounding_boxes_and_anomalies(
                    src_img, hek_bboxes, hek_polygons, anomaly_boxes, out_dir)
            iou = calculate_iou(hek_bboxes, anomaly_boxes)
            result = {
                "iou": iou,
                "image": src_img.name,
                "n_hek_events": len(hek_bboxes),
                "n_predicted_events": len(anomaly_boxes)
            }

            # TODO calculate overall iou
            ious.append(result)
        except:
            logger.exception(f"could not process image {src_img.name}")

    with open(iou_path, 'a', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=result_cols)
        writer.writerows(ious)
