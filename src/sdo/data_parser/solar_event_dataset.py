import xml.etree.ElementTree as ET
from pathlib import Path
from shapely.geometry import Polygon
import csv
import os
from os import path
import datetime as dt


def read_annotation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find('filename').text

    if len(list(root.iter('object'))) < 1:
        return filename, []

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        event_type = boxes.find("name").text

        list_with_all_boxes.append((event_type, [xmin, ymin, xmax, ymax]))

    return filename, list_with_all_boxes


# https://www.lmsal.com/hek/VOEvent_Spec.html
event_types = {
    "coronal_hole": "CH",
    "sunspot": "SS",
    "prominence": "PB"
}

# converts the labels to a more usable data format


def convert_csv(root_dir="~/Desktop/kasi_solar_event_dataset/test", target_dir="~/Desktop/kasi_solar_event_dataset"):
    target_dir = path.expanduser(target_dir)
    root_dir = path.expanduser(root_dir)
    if os.path.exists(target_dir) and not os.path.isdir(target_dir):
        raise ValueError(target_dir + " is not a directory")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    annotation_files = Path(root_dir).rglob(f'*.xml')
    csv_fieldnames = ['file_name', 'bbox',
                      'event_type', "wavelength", "instrument", "timestamp"]

    label_path = Path(target_dir) / "labels.csv"
    with open(label_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=csv_fieldnames)
        writer.writeheader()

    date_format = '%Y%m%d%H%M%S'
    for annotation_file in annotation_files:
        image_file, bboxes = read_annotation(annotation_file)
        instrument = image_file.split("SDO_")[1][0:3]
        timestamp = dt.datetime.strptime(image_file.split(
            "_")[0] + image_file.split("_")[1], date_format)

        if instrument == "HMI":
            wavelength = image_file.split("SDO_")[1][4:6]
        else:
            wavelength = image_file.split("SDO_")[1][4:7]

        # TODO convert pixelunit to arc sec
        for bbox_type in bboxes:
            event_type, bbox = bbox_type
            x_min_y_min = (bbox[0], bbox[1])
            x_max_y_min = (bbox[2], bbox[1])
            x_max_y_max = (bbox[2], bbox[3])
            x_min_y_max = (bbox[0], bbox[3])
            bb_poly = Polygon(
                [x_min_y_min, x_max_y_min, x_max_y_max, x_min_y_max])

            label = {}
            label["file_name"] = image_file
            label["bbox"] = bb_poly
            label["event_type"] = event_types[event_type]
            label["wavelength"] = wavelength
            label["instrument"] = instrument
            label["timestamp"] = timestamp.isoformat()
            with open(label_path, 'a', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=csv_fieldnames)
                writer.writerow(label)


convert_csv()
