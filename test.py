from ObjectDetectionModel import YOLOv8ObjDetectionModel
import cv2
import numpy as np

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
                if areas[i] < areas[j]:
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
    
    #point_on_src = cv2.perspectiveTransform(np.array([point_on_rect], dtype='float32'), M_inv)

    return transformed_image, M_inv


model_box = YOLOv8ObjDetectionModel(model_path='models/box_detection_model.pt')

model_box_2 = YOLOv8ObjDetectionModel(model_path='models/box_detection_model_2.pt')

model_person = YOLOv8ObjDetectionModel(model_path='models/person_detection_model.pt', classes=[0])

image = cv2.imread('person_test.png')

image_copy = image.copy()

result = model_person(image_copy, conf=0.1)
print(result)



#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image, M_inv = transform_quadrilateral_to_rectangle(image, np.array([[763.5471891007334,284.96698113207543],[769.8679438177145,430.3443396226415],[1108.028321176205,405.061320754717],[1119.089641930922,275.48584905660374]]))

results = model_box(image, conf=0.5)

results_2 = model_box(image, conf=0.4)



boxes = results.boxes.xyxy
prob = results.boxes

boxes_2 = results_2.boxes.xyxy
prob_2 = results_2.boxes

A = np.array(boxes.cpu())
B = np.array(boxes_2.cpu())

out_boxes = np.vstack((A, B))

print(out_boxes)

# remove rectangles, which are on other rectangles
out_boxes = remove_rectangles_based_on_intersections_and_area(out_boxes)

# draw rectangles
for i, box in enumerate(out_boxes):
    x1, y1, x2, y2 = list(box)
    label = 0
    points_1 = np.array([[[int(x1), int(y1)]]], dtype='float32')
    point_on_src_1 = cv2.perspectiveTransform(points_1, M_inv)
    points_2 = np.array([[[int(x2), int(y2)]]], dtype='float32')
    point_on_src_2 = cv2.perspectiveTransform(points_2, M_inv)
    image_copy = cv2.rectangle(image_copy, (int(point_on_src_1[0][0][0]), int(point_on_src_1[0][0][1])), (int(point_on_src_2[0][0][0]), int(point_on_src_2[0][0][1])),
                        (0, 255, 0), 2)

print(len(out_boxes))

cv2.imwrite(f"boxes_1_4.jpg", image_copy)
