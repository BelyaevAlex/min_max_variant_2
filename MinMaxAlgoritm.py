import cv2
import datetime
import uuid

import numpy as np

from ObjectDetectionModel import YOLOv8ObjDetectionModel
import logging
import colorlog
from dotenv import load_dotenv
from ImgExtract import ImageExtractor
import os
import json
from report import WorldReporter
import time


def create_logger():
    logger = logging.getLogger('HelloWorld_logger')
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'CRITICAL': 'bold_red,bg_white',
        }))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def area_of_rectangle(rect: list) -> int:
    x1, y1, x2, y2 = rect
    return abs(x2 - x1) * abs(y2 - y1)


def intersection_area(rect1: list, rect2: list) -> int:
    x1, y1, x2, y2 = rect1
    a1, b1, a2, b2 = rect2

    x_left = max(x1, a1)
    y_top = max(y1, b1)
    x_right = min(x2, a2)
    y_bottom = min(y2, b2)

    if x_right < x_left or y_bottom < y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)


def remove_rectangles_based_on_intersections_and_area(rectangles: list) -> list:
    coef_intersection = 0.51
    n = len(rectangles)
    areas = [area_of_rectangle(rect) for rect in rectangles]
    to_remove = set()

    for i in range(n):
        for j in range(i + 1, n):
            intersection = intersection_area(rectangles[i], rectangles[j])
            min_area = min(areas[i], areas[j])
            if intersection > coef_intersection * min_area:
                if areas[i] <= areas[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    filtered_rectangles = [rect for i, rect in enumerate(rectangles) if i not in to_remove]
    return filtered_rectangles


def find_corners(points):
    sorted_by_y = points[np.argsort(points[:, 1])]

    bottom_points = sorted_by_y[:2, :]
    top_points = sorted_by_y[2:, :]

    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    top_points = top_points[np.argsort(top_points[:, 0])]

    corners = np.array([bottom_points[0], bottom_points[1], top_points[1], top_points[0]])

    return corners


def transform_quadrilateral_to_rectangle(image, src_points):
    src_points = find_corners(src_points)
    width_a = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + ((src_points[2][1] - src_points[3][1]) ** 2))
    width_b = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + ((src_points[1][1] - src_points[0][1]) ** 2))
    width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((src_points[1][0] - src_points[2][0]) ** 2) + ((src_points[1][1] - src_points[2][1]) ** 2))
    height_b = np.sqrt(((src_points[0][0] - src_points[3][0]) ** 2) + ((src_points[0][1] - src_points[3][1]) ** 2))
    height = max(int(height_a), int(height_b))

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    matrix = cv2.getPerspectiveTransform(src_points.astype('float32'), dst_points)

    transformed_image = cv2.warpPerspective(image, matrix, (width, height))

    M_inv = cv2.getPerspectiveTransform(dst_points, src_points.astype('float32'))

    return transformed_image, M_inv


logg = create_logger()
myColor = 0
logg.info("Project was started")

# get environments
if os.environ.get("server_url") is None:
    load_dotenv(".env")
extra: str = os.environ.get("extra")
extra = json.loads(extra)[0]
zones = extra.get("areas")
logg.info(f"Was founded {len(zones)} areas")
server_url = os.environ.get("server_url")
camera_url = os.environ.get("camera_url")
camera_ip = os.environ.get("camera_ip")
folder = os.environ.get("folder")

# init classes
report = WorldReporter(server_url, folder, logg)
model_box = YOLOv8ObjDetectionModel(model_path='models/box_detection_model.pt')
model_box_2 = YOLOv8ObjDetectionModel(model_path='models/box_detection_model_2.pt')
model_person = YOLOv8ObjDetectionModel(model_path='models/person_detection_model.pt', classes=[0])
image_extr = ImageExtractor(camera_url, logg)
logg.info("All env variables and models was loaded successfully")

# status 0 - boxes already recounted, check for person
# status 1 - person on image
# status 2 - person leave, but boxes not recount. Recounting boxes
status = 2
# make sure that person leave
accept = 0

while True:
    logg.info(f"Try to load image")
    image = image_extr.get_image()
    image_color_changed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logg.info(f"Image was loaded successfully")

    predict_person = model_person(image_color_changed, conf=0.1)

    if len(predict_person.boxes.xyxy) > 0 and status == 0:
        status = 1
        logg.info(f"Person was founded")
        accept = 0
    elif status == 1:
        accept += 1
        logg.info(f"Person isn't in image already {accept} seconds")
        if accept >= 5:
            status = 2
            logg.info(f"Person leave")
            accept = 0

    if status == 2:
        logg.info("Start recounting boxes")
        final_report = []
        start_track = datetime.datetime.now()
        for zone in zones:
            # get four coordinates of zone with boxes
            values = zone["coords"][0]
            image_copy = image_color_changed.copy()
            x1_zone, y1_zone, x2_zone, y2_zone, x3_zone, y3_zone, x4_zone, y4_zone = int(values["x1"]), int(
                values["y1"]), int(values["x2"]), int(
                values["y2"]), int(values["x3"]), int(values["y3"]), int(values["x4"]), int(values["y4"])

            img, M_inv = transform_quadrilateral_to_rectangle(image_copy, [(x1_zone, y1_zone), (x2_zone, y2_zone),
                                                                           (x3_zone, y3_zone), (x4_zone, y4_zone)])

            # make prediction
            results = model_box(img, conf=0.5)
            result_2 = model_box_2(img, conf=0.4)

            # get data
            boxes_1 = results.boxes.xyxy.cpu()
            boxes_2 = result_2.boxes.xyxy.cpu()
            boxes = np.vstack((boxes_1, boxes_2))

            # remove rectangles, which are on other rectangles
            boxes = remove_rectangles_based_on_intersections_and_area(boxes)

            # draw zone on main image
            image_with_zone = cv2.rectangle(image_copy, (x1_zone, y1_zone), (x2_zone, y2_zone), (255, 0, 255), 2)
            img = cv2.putText(image_with_zone, zone['itemName'], (int(x1_zone), int(y1_zone) - 5),
                              cv2.FONT_HERSHEY_COMPLEX,
                              0.5, (255, 0, 255), 1)

            final_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = list(box)
                if x2 - x1 > 10 and y2 - y1 > 10:
                    final_boxes.append(box)

            # draw rectangles
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = list(box)
                label = 0
                points_1 = np.array([[[int(x1), int(y1)]]], dtype='float32')
                point_on_src_1 = cv2.perspectiveTransform(points_1, M_inv)
                points_2 = np.array([[[int(x2), int(y2)]]], dtype='float32')
                point_on_src_2 = cv2.perspectiveTransform(points_2, M_inv)
                img = cv2.rectangle(image_copy, (int(point_on_src_1[0][0][0]), int(point_on_src_1[0][0][1])),
                                    (int(point_on_src_2[0][0][0]), int(point_on_src_2[0][0][1])),
                                    (0, 255, 0), 2)
                img = cv2.putText(img, str(i + 1),
                                  (int(point_on_src_1[0][0][0]) + 4, int(point_on_src_1[0][0][1]) + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 255, 255), 1)
            logg.info(f"In zone {zone['itemName']} was founded {len(boxes)} boxes")

            name_file = uuid.uuid4()
            file_path = os.path.join(folder, f"{name_file}.jpg")
            cv2.imwrite(file_path, img)

            # send report
            final_report.append(report.create_item_report(file_path, int(zone['itemId']), len(boxes)))
        end_track = datetime.datetime.now()
        report.send_report(str(start_track), str(end_track), camera_ip, final_report)
        logg.info(f"Boxes was recounted")
        status = 0
    time.sleep(1)
