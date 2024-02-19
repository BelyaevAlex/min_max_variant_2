import cv2
import datetime
import uuid
from ObjectDetectionModel import YOLOv8ObjDetectionModel
import logging
import colorlog
from dotenv import load_dotenv
from ImgExtract import ImageExtractor
import os
import json
from report import WorldReporter


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
    coef_intersection = 0.7
    n = len(rectangles)
    areas = [area_of_rectangle(rect) for rect in rectangles]
    to_remove = set()

    for i in range(n):
        for j in range(i + 1, n):
            intersection = intersection_area(rectangles[i], rectangles[j])
            min_area = min(areas[i], areas[j])
            if intersection > coef_intersection * min_area:
                if areas[i] > areas[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    filtered_rectangles = [rect for i, rect in enumerate(rectangles) if i not in to_remove]
    return filtered_rectangles


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
print(os.environ.get("extra"))
server_url = os.environ.get("server_url")
camera_url = os.environ.get("camera_url")
camera_ip = os.environ.get("camera_ip")
folder = os.environ.get("folder")

# init classes
report = WorldReporter(server_url, folder, logg)
model_box = YOLOv8ObjDetectionModel(model_path='models/box_detection_model.pt')
model_person = YOLOv8ObjDetectionModel(model_path='models/person_detection_model.pt')
image_extr = ImageExtractor(camera_url, logg)
logg.info("All env variables and models was loaded successfully")

# status 0 - boxes already recounted, check for person
# status 1 - person on image
# status 2 - person leave, but boxes not recount. Recounting boxes
status = 2

while True:
    logg.info(f"Try to load image")
    image = image_extr.get_image()
    logg.info(f"Image was loaded successfully")

    predict_person = model_person(image, conf=0.5)

    if len(predict_person.boxes.xyxy) > 0 and status == 0:
        status = 1
        logg.info(f"Person was founded")
    elif status == 1:
        status = 2
        logg.info(f"Person leave")

    if status == 2:
        logg.info("Start recounting boxes")
        final_report = []
        start_track = datetime.datetime.now()
        for zone in zones:
            # get four coordinates of zone with boxes
            values = zone["coords"][0]
            x1, y1, x2, y2 = int(values["x1"]), int(values["y1"]), int(values["x2"]), int(values["y2"])

            img = image[y1:y2, x1:x2]

            # make prediction
            results = model_box(img, conf=0.55)

            # get data
            boxes = results.boxes.xyxy
            prob = results.boxes

            # remove rectangles, which are on other rectangles
            boxes = remove_rectangles_based_on_intersections_and_area(boxes)

            # draw rectangles
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = list(box)
                label = 0
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                img = cv2.putText(img, str(prob[i].conf[0].item())[:4], (int(x1) + 4, int(y1) + 15),
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
