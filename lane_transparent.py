import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array                   #type:ignore
from tensorflow.keras.models import load_model                                            #type:ignore
import os
import pygame
import torch
import math
import psutil
import os
import time

# def memory_usage():
#     # Get the process ID (PID) of the current Python script
#     process = psutil.Process(os.getpid())
#     # Return the memory usage in MB
#     return process.memory_info().rss / (1024 * 1024)
# print("memory usagge before:",memory_usage())
# Load the saved model
model = load_model('8batch20epoch.h5')
# Optional: Display model summary
model.summary()
pygame.mixer.init()
# Load the YOLOv5 model for object detection
yolo_model_path = 'yolov56/runs/train/exp3/weights/best.pt'
yolo_model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
# Function to detect objects using YOLOv5
def detect_objects(image, model):
    results = model(image)
    return results
# Function to plot YOLOv5 detection results on an image
def plot_results(image, results, class_names, conf_threshold=0.5):
    for result in results.xyxy[0].cpu().numpy():
        xmin, ymin, xmax, ymax, conf, cls = result
        if conf >= conf_threshold:  # Only display if confidence is above the threshold
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(image, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# Hyperparameters
img_height, img_width = 128, 128
display_height, display_width = 512, 512  # Adjust the size for displaying images
def roi(black_image_resized_bgr):
    # Draw a vertical line 20 pixels from the bottom in the middle of the black image in blue color
    line_x = black_image_resized_bgr.shape[1] // 2
    line_y_start = black_image_resized_bgr.shape[0] - 20
    line_y_end = black_image_resized_bgr.shape[0]
    cv2.line(black_image_resized_bgr, (line_x, line_y_start), (line_x, line_y_end), (255, 0, 0), 1)
    # Check if all rows to the left or right of the line are black
    left_of_line = black_image_resized_bgr[:, :line_x]
    right_of_line = black_image_resized_bgr[:, line_x:]
    return left_of_line, right_of_line
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

def detect_image_with_lines(frame):
    results = yolo_model(frame)  # Perform object detection
    detections = results.pandas().xyxy[0]  # Extract detections
    relevant_classes = {'car', 'truck', 'bus', 'bicycle', 'motorbike', 'person'}
    height, width, _ = frame.shape
    removed_area = np.copy(frame)
    bottom_line_coordinates = []
    num_bounding_boxes = 0  # Initialize the counter for bounding boxes

    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls, name = row
        if name in relevant_classes:  # Filter out specific classes if needed
            num_bounding_boxes += 1  # Increment the counter for each bounding box
            
            # Calculate bottom line coordinates (x-coordinates from left to right on the bottom line of the bounding box)
            bottom_line_x_coordinates = list(range(int(x1), int(x2) + 1))
            bottom_line_y_coordinate = int(y2)
            bottom_line_coords = [(x, bottom_line_y_coordinate) for x in bottom_line_x_coordinates]
            bottom_line_coordinates.append(bottom_line_coords)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{name} {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Remove the detected area from the copy of the frame
            cv2.rectangle(removed_area, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)

    return frame, removed_area, bottom_line_coordinates, num_bounding_boxes

def is_touching_magenta_line(coords, magenta_line):
    x1, y1 = coords[0], coords[1]
    x2, y2 = magenta_line[0][0], magenta_line[0][1]
    x3, y3 = magenta_line[1][0], magenta_line[1][1]

    dist = abs((y2 - y3) * x1 + (x3 - x2) * y1 + (x2 * y3 - x3 * y2)) / np.sqrt((y2 - y3) ** 2 + (x3 - x2) ** 2)
    if dist < 5:  # Adjust this threshold as needed
        return True
    return False


def extend_line_to_bottom(point1, point2, height):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    if point2[1] < height:
        extended_y = height
        extended_x = int(point1[0] + (extended_y - point1[1]) / slope)
        return point2, (extended_x, extended_y)
    return None
def extend_line_downwards(point1, point2, width, height):
    if point1[1] == point2[1]:  # Check if the line is horizontal
        extended_y = height - 1
        extended_x = point1[0]
    else:
        slope = (point2[0] - point1[0]) / (point2[1] - point1[1])
        if slope == 0:
            extended_x = point1[0]
            extended_y = height - 1
        else:
            extended_y = height - 1
            extended_x = int(point1[0] + (extended_y - point1[1]) * slope)
            if extended_x < 0:  # Line intersects the left boundary
                extended_x = 0
                extended_y = int(point1[1] + (0 - point1[0]) / slope)
            elif extended_x > width - 1:  # Line intersects the right boundary
                extended_x = width - 1
                extended_y = int(point1[1] + (width - 1 - point1[0]) / slope)
    return (extended_x, extended_y)

def is_point_on_line_segment(p1, p2, point, threshold=5):
    """ Check if the point lies on the line segment between p1 and p2 """
    line_mag = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    u = ((point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (p2[1] - p1[1])) / float(line_mag**2)
    if u < 0.0 or u > 1.0:
        return False
    intersection = (int(p1[0] + u * (p2[0] - p1[0])), int(p1[1] + u * (p2[1] - p1[1])))
    dist = np.sqrt((intersection[0] - point[0])**2 + (intersection[1] - point[1])**2)
    return dist < threshold

def calculate_angle(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def is_line_touching_bottom_coordinates(line_start, line_end, bottom_coords):
    for coord_set in bottom_coords:
        for (x, y) in coord_set:
            if min(line_start[0], line_end[0]) <= x <= max(line_start[0], line_end[0]) and min(line_start[1], line_end[1]) <= y <= max(line_start[1], line_end[1]):
                line_slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0]) if line_end[0] != line_start[0] else float('inf')
                y_intercept = line_start[1] - line_slope * line_start[0]
                if abs(y - (line_slope * x + y_intercept)) < 1e-2:
                    return True
    return False

def calcu_slope(p1, p2):
    if p1[0] == p2[0]:  # vertical line
        return float('inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def calculate_slope(point1, point2):
    if point2[0] - point1[0] != 0:
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        return float('inf')  # Avoid division by zero

def calculate_yellow_line_coordinates(point1, point2, y_extended):
    try:
        if point1[1] == point2[1]:  # Check if the line is horizontal
            yellow_x = point1[0]
            return (yellow_x, y_extended)
        else:
            slope = (point2[0] - point1[0]) / (point2[1] - point1[1])
            yellow_x = int(point1[0] + (y_extended - point1[1]) * slope)
            return (yellow_x, y_extended)
    except Exception as e:
        print(f"Error calculating yellow line coordinates: {e}")
        return (None, None)


def draw_left_right_dots_and_top_line(removed_area, frame_resized_for_display, bottom_line_coordinates):
    height, width, _ = removed_area.shape
    top_line_row = None
    
    for y in range(height):
        if np.any(removed_area[y, :] != 0):
            top_line_row = y
            break

    line_length_10th_row = None
    midpoint_10th_row = None

    if top_line_row is not None:
        leftmost_non_zero_x_10th = None
        rightmost_non_zero_x_10th = None
        
        tenth_row_below = top_line_row + 10
        if tenth_row_below < height:
            for x in range(width):
                if np.any(removed_area[tenth_row_below, x] != 0):
                    leftmost_non_zero_x_10th = x
                    break
            for x in range(width - 1, -1, -1):
                if np.any(removed_area[tenth_row_below, x] != 0):
                    rightmost_non_zero_x_10th = x
                    break
            
            if leftmost_non_zero_x_10th is not None and rightmost_non_zero_x_10th is not None:
                cv2.line(frame_resized_for_display, (leftmost_non_zero_x_10th, tenth_row_below), 
                         (rightmost_non_zero_x_10th, tenth_row_below), (0, 0, 255), 2)
                line_length_10th_row = rightmost_non_zero_x_10th - leftmost_non_zero_x_10th
                midpoint_10th_row = ((leftmost_non_zero_x_10th + rightmost_non_zero_x_10th) // 2, tenth_row_below)
    
    left_dot = None
    right_dot = None
    for y in range(height):
        if np.any(removed_area[y, 0] != 0):
            left_dot = (0, y)
            break
    for y in range(height):
        if np.any(removed_area[y, width - 1] != 0):
            right_dot = (width - 1, y)
            break
    if left_dot is not None:
        cv2.circle(frame_resized_for_display, left_dot, 5, (0, 0, 255), -1)  
    if right_dot is not None:
        cv2.circle(frame_resized_for_display, right_dot, 5, (0, 0, 255), -1)
    if left_dot and right_dot:
        left_distance = height - left_dot[1]
        right_distance = height - right_dot[1]
        farther_dot = left_dot if left_distance > right_distance else right_dot
        if farther_dot:
            if farther_dot == right_dot:
                y = right_dot[1]
                leftmost_non_zero_x = None
                for x in range(width):
                    if np.any(removed_area[y, x] != 0):
                        leftmost_non_zero_x = x
                        break
                if leftmost_non_zero_x is not None:
                    if leftmost_non_zero_x_10th is not None and rightmost_non_zero_x_10th is not None:
                        def calculate_division_points(left_x, right_x, y):
                            segment_length = (right_x - left_x) / 3
                            point1_x = int(left_x + segment_length)
                            point2_x = int(left_x + 2 * segment_length)
                            return (point1_x, y), (point2_x, y)

                        point1_green, point2_green = calculate_division_points(leftmost_non_zero_x, right_dot[0], y)
                        point1_10th, point2_10th = calculate_division_points(leftmost_non_zero_x_10th, rightmost_non_zero_x_10th, tenth_row_below)

                        cv2.line(frame_resized_for_display, point1_green, point1_10th, (0, 255, 255), 2)
                        cv2.line(frame_resized_for_display, point2_green, point2_10th, (0, 255, 255), 2)

                        midpoint_green = ((leftmost_non_zero_x + point1_green[0]) // 2, (y + point1_green[1]) // 2)
                        midpoint_10th = ((leftmost_non_zero_x_10th + point1_10th[0]) // 2, (tenth_row_below + point1_10th[1]) // 2)

                        cv2.line(frame_resized_for_display, midpoint_green, midpoint_10th, (0, 255, 0), 2)

                        yellow_line1 = extend_line_to_bottom(point1_green, point1_10th, height)
                        yellow_line2 = extend_line_to_bottom(point2_green, point2_10th, height)
                        magenta_line = extend_line_to_bottom(midpoint_green, midpoint_10th, height)

                        if yellow_line1:
                            cv2.line(frame_resized_for_display, yellow_line1[0], yellow_line1[1], (0, 255, 255), 2)
                        if yellow_line2:
                            cv2.line(frame_resized_for_display, yellow_line2[0], yellow_line2[1], (0, 255, 255), 2)
                        if magenta_line:
                            cv2.line(frame_resized_for_display, magenta_line[0], magenta_line[1], (0, 255, 0), 2)

                        cv2.line(frame_resized_for_display, (leftmost_non_zero_x, y), (leftmost_non_zero_x_10th, tenth_row_below), (0, 0, 255), 2)
                        cv2.line(frame_resized_for_display, (right_dot[0], y), (rightmost_non_zero_x_10th, tenth_row_below), (0, 0, 255), 2)
                        
                        intersection_found = False
                        y_intersect = None
                        for bottom_line in bottom_line_coordinates:
                            for coords in bottom_line:
                                if magenta_line and is_touching_magenta_line(coords, magenta_line):
                                    intersection_found = True
                                    y_intersect = coords[1]
                                    break
                            if intersection_found:
                                break
                        if magenta_line:
                            angle = np.arctan2(magenta_line[1][1] - magenta_line[0][1], magenta_line[1][0] - magenta_line[0][0]) * 180 / np.pi
                        if intersection_found and y_intersect is not None:
                            if angle > 96:
                                cv2.putText(frame_resized_for_display, "vehicle on left", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            elif angle < 96:
                                cv2.putText(frame_resized_for_display, "vehicle ahead", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)                            
                            overlay = frame_resized_for_display.copy()
                            for y in range(y_intersect, height):
                                leftmost_non_zero_x_row = None
                                for x in range(width):
                                    if np.any(removed_area[y, x] != 0):
                                        leftmost_non_zero_x_row = x
                                        break
                                if leftmost_non_zero_x_row is not None:
                                    yellow_x, _ = calculate_yellow_line_coordinates(point1_green, point1_10th, y)
                                    cv2.line(overlay, (leftmost_non_zero_x_row, y), (yellow_x, y), (0, 255, 0), 2)
                            alpha = 0.2
                            cv2.addWeighted(overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
                        else:
                            if angle > 96:
                                cv2.putText(frame_resized_for_display, "keep left", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            overlay = frame_resized_for_display.copy()
                            for y in range(tenth_row_below, height):
                                leftmost_non_zero_x_row = None
                                for x in range(width):
                                    if np.any(removed_area[y, x] != 0):
                                        leftmost_non_zero_x_row = x
                                        break
                                if leftmost_non_zero_x_row is not None:
                                    yellow_x, _ = calculate_yellow_line_coordinates(point1_green, point1_10th, y)
                                    cv2.line(overlay, (leftmost_non_zero_x_row, y), (yellow_x, y), (0, 255, 0), 2)

                            alpha = 0.2
                            cv2.addWeighted(overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
            elif farther_dot == left_dot:
                y = left_dot[1]
                rightmost_non_zero_x = None
                for x in range(width - 1, -1, -1):
                    if np.any(removed_area[y, x] != 0):
                        rightmost_non_zero_x = x
                        break
                if rightmost_non_zero_x is not None:
                #     # cv2.line(frame_resized_for_display, left_dot, (rightmost_non_zero_x, y), (0, 0, 255), 2)  
                    if leftmost_non_zero_x_10th is not None and rightmost_non_zero_x_10th is not None:
                        def calculate_division_points(left_x, right_x, y):
                            segment_length = (right_x - left_x) / 3
                            point1_x = int(left_x + segment_length)
                            point2_x = int(left_x + 2 * segment_length)
                            return (point1_x, y), (point2_x, y)

                        point1_green, point2_green = calculate_division_points(left_dot[0], rightmost_non_zero_x, y)
                        point1_10th, point2_10th = calculate_division_points(leftmost_non_zero_x_10th, rightmost_non_zero_x_10th, tenth_row_below)

                        cv2.line(frame_resized_for_display, point1_green, point1_10th, (0, 255, 255), 2)
                        cv2.line(frame_resized_for_display, point2_green, point2_10th, (0, 255, 255), 2)

                        midpoint_green = ((left_dot[0] + point1_green[0]) // 2, (y + point1_green[1]) // 2)
                        midpoint_10th = ((leftmost_non_zero_x_10th + point1_10th[0]) // 2, (tenth_row_below + point1_10th[1]) // 2)

                        cv2.line(frame_resized_for_display, midpoint_green, midpoint_10th, (0, 255, 0), 2)

                        yellow_line1 = extend_line_to_bottom(point1_green, point1_10th, height)
                        yellow_line2 = extend_line_to_bottom(point2_green, point2_10th, height)
                        magenta_line = extend_line_to_bottom(midpoint_green, midpoint_10th, height)

                        if yellow_line1:
                            cv2.line(frame_resized_for_display, yellow_line1[0], yellow_line1[1], (0, 255, 255), 2)
                        if yellow_line2:
                            cv2.line(frame_resized_for_display, yellow_line2[0], yellow_line2[1], (0, 255, 255), 2)
                        if magenta_line:
                            cv2.line(frame_resized_for_display, magenta_line[0], magenta_line[1], (0, 255, 0), 2)

                        cv2.line(frame_resized_for_display, (left_dot[0], y), (leftmost_non_zero_x_10th, tenth_row_below), (0, 0, 255), 2)
                        cv2.line(frame_resized_for_display, (rightmost_non_zero_x, y), (rightmost_non_zero_x_10th, tenth_row_below), (0, 0, 255), 2)
                        
                        intersection_found = False
                        y_intersect = None
                        for bottom_line in bottom_line_coordinates:
                            for coords in bottom_line:
                                if magenta_line and is_touching_magenta_line(coords, magenta_line):
                                    intersection_found = True
                                    y_intersect = coords[1]
                                    break
                            if intersection_found:
                                break
                        if magenta_line:
                            angle = np.arctan2(magenta_line[1][1] - magenta_line[0][1], magenta_line[1][0] - magenta_line[0][0]) * 180 / np.pi
                        if intersection_found and y_intersect is not None:
                            overlay = frame_resized_for_display.copy()
                            if angle > 96:
                                cv2.putText(frame_resized_for_display, "vehicle on left", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            elif angle < 96:
                                cv2.putText(frame_resized_for_display, "vehicle ahead", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)                            
                            for y in range(y_intersect, height):
                                leftmost_non_zero_x_row = None
                                for x in range(width):
                                    if np.any(removed_area[y, x] != 0):
                                        leftmost_non_zero_x_row = x
                                        break
                                if leftmost_non_zero_x_row is not None:
                                    yellow_x, _ = calculate_yellow_line_coordinates(point1_green, point1_10th, y)
                                    cv2.line(overlay, (leftmost_non_zero_x_row, y), (yellow_x, y), (0, 255, 0), 2)
                            alpha = 0.2
                            cv2.addWeighted(overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
                        else:
                            overlay = frame_resized_for_display.copy()
                            if angle > 96:
                                cv2.putText(frame_resized_for_display, "keep left", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            for y in range(tenth_row_below, height):
                                leftmost_non_zero_x_row = None
                                for x in range(width):
                                    if np.any(removed_area[y, x] != 0):
                                        leftmost_non_zero_x_row = x
                                        break
                                if leftmost_non_zero_x_row is not None:
                                    yellow_x, _ = calculate_yellow_line_coordinates(point1_green, point1_10th, y)
                                    cv2.line(overlay, (leftmost_non_zero_x_row, y), (yellow_x, y), (0, 255, 0), 2)
                            alpha = 0.2
                            cv2.addWeighted(overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
    # elif (left_dot and right_dot) and ((height - left_dot[1] < 100) or (height - right_dot[1] < 100)):
    #     pass
    elif left_dot and not right_dot:
        cv2.putText(frame_resized_for_display, "keep left", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load('keep_left.mp3')
            pygame.mixer.music.play()
        return frame_resized_for_display
    elif right_dot and not left_dot:
        y = right_dot[1]
        leftmost_non_zero_x = None
        for x in range(width):
            if np.any(removed_area[y, x] != 0):
                leftmost_non_zero_x = x
                break
        if leftmost_non_zero_x is not None:
            if leftmost_non_zero_x_10th is not None:
                cv2.line(frame_resized_for_display, (leftmost_non_zero_x_10th, tenth_row_below), (leftmost_non_zero_x, y), (0, 0, 255), 2)
            if rightmost_non_zero_x_10th is not None:
                cv2.line(frame_resized_for_display, (rightmost_non_zero_x_10th, tenth_row_below), right_dot, (0, 0, 255), 2)

            midpoint_new_line = ((leftmost_non_zero_x + right_dot[0]) // 2, y)

            if midpoint_10th_row:
                cv2.line(frame_resized_for_display, midpoint_10th_row, midpoint_new_line, (0, 255, 255), 2)

                extended_yellow_line = extend_line_downwards(midpoint_10th_row, midpoint_new_line, width, height)
                cv2.line(frame_resized_for_display, midpoint_new_line, extended_yellow_line, (0, 255, 255), 2)

            if leftmost_non_zero_x_10th and midpoint_10th_row:
                mid_left_10th_to_mid_10th = ((leftmost_non_zero_x_10th + midpoint_10th_row[0]) // 2, tenth_row_below)
            if leftmost_non_zero_x and midpoint_new_line:
                mid_left_new_to_mid_new = ((leftmost_non_zero_x + midpoint_new_line[0]) // 2, y)

            if leftmost_non_zero_x_10th and midpoint_10th_row and leftmost_non_zero_x and midpoint_new_line:
                cv2.line(frame_resized_for_display, mid_left_10th_to_mid_10th, mid_left_new_to_mid_new, (0, 255, 0), 2)

                extended_cyan_line = extend_line_downwards(mid_left_10th_to_mid_10th, mid_left_new_to_mid_new, width, height)
                cv2.line(frame_resized_for_display, mid_left_new_to_mid_new, extended_cyan_line, (0, 255, 0), 2)

            if midpoint_10th_row and extended_yellow_line:
                angle_yellow_red = abs(calculate_angle(midpoint_10th_row, extended_yellow_line))

            if line_length_10th_row is not None and line_length_10th_row <= 65:
                green_overlay = frame_resized_for_display.copy()

                for row in range(tenth_row_below, height):
                    if midpoint_new_line and not is_line_touching_bottom_coordinates(midpoint_new_line, extended_yellow_line, bottom_line_coordinates):
                        if angle_yellow_red > 80:
                            cv2.putText(frame_resized_for_display, "vehicle on left", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame_resized_for_display, "keep left", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        leftmost_non_zero_x = None
                        for x in range(width):
                            if np.any(removed_area[row, x] != 0):
                                leftmost_non_zero_x = x
                                break

                        if leftmost_non_zero_x is not None:
                            y_extended = row
                            yellow_x, _ = calculate_yellow_line_coordinates(midpoint_10th_row, midpoint_new_line, y_extended)
                            cv2.line(green_overlay, (leftmost_non_zero_x, y_extended), (yellow_x, y_extended), (0, 255, 0), 2)

                    else:
                        if angle_yellow_red > 80:
                            cv2.putText(frame_resized_for_display, "keep left", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            for coord_set in bottom_line_coordinates:
                                for (x, y) in coord_set:
                                    if y == row:
                                        leftmost_non_zero_x = None
                                        for x in range(width):
                                            if np.any(removed_area[y, x] != 0):
                                                leftmost_non_zero_x = x
                                                break
                                        if leftmost_non_zero_x is not None:
                                            y_extended = y
                                            yellow_x, _ = calculate_yellow_line_coordinates(midpoint_10th_row, midpoint_new_line, y_extended)
                                            cv2.line(green_overlay, (leftmost_non_zero_x, y_extended), (yellow_x, y_extended), (0, 255, 0), 2)
                alpha = 0.2
                cv2.addWeighted(green_overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
            elif line_length_10th_row is not None and line_length_10th_row >= 65:
                # Transparent line drawing
                transparent_overlay = frame_resized_for_display.copy()
                for row in range(tenth_row_below, height):
                    leftmost_non_zero_x = None
                    rightmost_non_zero_x = None
                    for x in range(width):
                        if np.any(removed_area[row, x] != 0):
                            if leftmost_non_zero_x is None:
                                leftmost_non_zero_x = x
                            rightmost_non_zero_x = x
                    if leftmost_non_zero_x is not None and rightmost_non_zero_x is not None:
                        midpoint_of_row = ((leftmost_non_zero_x + rightmost_non_zero_x) // 2, row)
                        cv2.line(transparent_overlay, (leftmost_non_zero_x, row), midpoint_of_row, (0, 255, 0), 2)
                
                alpha = 0.2
                cv2.addWeighted(transparent_overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)

    else:
        # Draw a line from the leftmost non-zero pixel to the rightmost non-zero pixel of the last row
        last_row = height - 1
        leftmost_last_row = None
        rightmost_last_row = None
        for x in range(width):
            if np.any(removed_area[last_row, x] != 0):
                leftmost_last_row = x
                break
        for x in range(width - 1, -1, -1):
            if np.any(removed_area[last_row, x] != 0):
                rightmost_last_row = x
                break
        if leftmost_last_row is not None and rightmost_last_row is not None:
            cv2.line(frame_resized_for_display, (leftmost_last_row, last_row), 
                    (rightmost_last_row, last_row), (0, 0, 255), 2)
            
            # Calculate the length of the line
            line_length = abs(rightmost_last_row - leftmost_last_row)
            
            # Check if the length is greater than half of the image width
            if line_length <= width / 2:
                # Draw lines connecting 10th row to the drawn line
                if tenth_row_below < height and midpoint_10th_row is not None:
                    cv2.line(frame_resized_for_display, (leftmost_non_zero_x_10th, tenth_row_below), 
                            (leftmost_last_row, last_row), (0, 0, 255), 2)
                    cv2.line(frame_resized_for_display, (rightmost_non_zero_x_10th, tenth_row_below), 
                            (rightmost_last_row, last_row), (0, 0, 255), 2)
                    
                    # Calculate midpoints
                    midpoint_last_row = ((leftmost_last_row + rightmost_last_row) // 2, last_row)
                    
                    # Draw a line between midpoints
                    cv2.line(frame_resized_for_display, midpoint_10th_row, midpoint_last_row, (0, 255, 255), 2)
                    
                    # Calculate new midpoints
                    mid_left_10th_to_mid_10th = ((leftmost_non_zero_x_10th + midpoint_10th_row[0]) // 2, tenth_row_below)
                    mid_left_last_to_mid_last = ((leftmost_last_row + midpoint_last_row[0]) // 2, last_row)
                    
                    # Draw a line between these new midpoints
                    cv2.line(frame_resized_for_display, mid_left_10th_to_mid_10th, mid_left_last_to_mid_last, (0, 255, 0), 2)

                    # Check the angle between the cyan line and the bottom green line
                    slope_cyan = calculate_slope(mid_left_10th_to_mid_10th, mid_left_last_to_mid_last)
                    slope_green = calculate_slope((leftmost_last_row, last_row), (rightmost_last_row, last_row))
                    
                    angle_radians = math.atan(abs((slope_cyan - slope_green) / (1 + slope_cyan * slope_green)))
                    angle_degrees = math.degrees(angle_radians)
                    
                    # if angle_degrees < 86:
                    
                    green_overlay = frame_resized_for_display.copy()
                    intersection_found = False

                    for row in range(tenth_row_below, height):
                        if midpoint_10th_row and midpoint_last_row:
                            for coord_set in bottom_line_coordinates:
                                for (x, y) in coord_set:
                                    if y == row:
                                        intersection_found = True
                                        break
                            if intersection_found:
                                break
                        
                    if line_length_10th_row is not None and line_length_10th_row <= 55:
                        leftmost_non_zero_x = None
                        for x in range(width):
                            if np.any(removed_area[row, x] != 0):
                                leftmost_non_zero_x = x
                                break
                        if leftmost_non_zero_x is not None:
                            y_extended = row
                            yellow_x = midpoint_10th_row[0] + (row - midpoint_10th_row[1]) * (midpoint_last_row[0] - midpoint_10th_row[0]) / (midpoint_last_row[1] - midpoint_10th_row[1])
                            cv2.line(green_overlay, (leftmost_non_zero_x, y_extended), (int(yellow_x), y_extended), (0, 255, 0), 2)
                        if intersection_found:
                            if angle_degrees < 86:
                                cv2.putText(frame_resized_for_display, "vehicle on left", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame_resized_for_display, "vehicle ahead", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            for row in range(row, height):
                                leftmost_non_zero_x = None
                                for x in range(width):
                                    if np.any(removed_area[row, x] != 0):
                                        leftmost_non_zero_x = x
                                        break
                                if leftmost_non_zero_x is not None:
                                    y_extended = row
                                    yellow_x = midpoint_10th_row[0] + (row - midpoint_10th_row[1]) * (midpoint_last_row[0] - midpoint_10th_row[0]) / (midpoint_last_row[1] - midpoint_10th_row[1])
                                    cv2.line(green_overlay, (leftmost_non_zero_x, y_extended), (int(yellow_x), y_extended), (0, 255, 0), 2)

                        else:
                            if angle_degrees < 86:
                                cv2.putText(frame_resized_for_display, "keep left", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            for row in range(tenth_row_below, height):
                                leftmost_non_zero_x = None
                                for x in range(width):
                                    if np.any(removed_area[row, x] != 0):
                                        leftmost_non_zero_x = x
                                        break
                                if leftmost_non_zero_x is not None:
                                    y_extended = row
                                    yellow_x = midpoint_10th_row[0] + (row - midpoint_10th_row[1]) * (midpoint_last_row[0] - midpoint_10th_row[0]) / (midpoint_last_row[1] - midpoint_10th_row[1])
                                    cv2.line(green_overlay, (leftmost_non_zero_x, y_extended), (int(yellow_x), y_extended), (0, 255, 0), 2)

                        alpha = 0.2
                        cv2.addWeighted(green_overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
                    if line_length_10th_row is not None and line_length_10th_row >= 55:
                        leftmost_non_zero_x = None
                        rightmost_non_zero_x = None
                        for x in range(width):
                            if np.any(removed_area[row, x] != 0):
                                leftmost_non_zero_x = x
                                break
                        for x in range(width - 1, -1, -1):
                            if np.any(removed_area[row, x] != 0):
                                rightmost_non_zero_x = x
                                break
                        if leftmost_non_zero_x is not None and rightmost_non_zero_x is not None:
                            midpoint_row = (leftmost_non_zero_x + rightmost_non_zero_x) // 2
                            cv2.line(green_overlay, (leftmost_non_zero_x, row), (midpoint_row, row), (0, 255, 0), 2)
                        if intersection_found:
                            for row in range(row, height):
                                leftmost_non_zero_x = None
                                rightmost_non_zero_x = None
                                for x in range(width):
                                    if np.any(removed_area[row, x] != 0):
                                        leftmost_non_zero_x = x
                                        break
                                for x in range(width - 1, -1, -1):
                                    if np.any(removed_area[row, x] != 0):
                                        rightmost_non_zero_x = x
                                        break
                                if leftmost_non_zero_x is not None and rightmost_non_zero_x is not None:
                                    midpoint_row = (leftmost_non_zero_x + rightmost_non_zero_x) // 2
                                    cv2.line(green_overlay, (leftmost_non_zero_x, row), (midpoint_row, row), (0, 255, 0), 2)
                        else:
                            for row in range(tenth_row_below, height):
                                leftmost_non_zero_x = None
                                rightmost_non_zero_x = None
                                for x in range(width):
                                    if np.any(removed_area[row, x] != 0):
                                        leftmost_non_zero_x = x
                                        break
                                for x in range(width - 1, -1, -1):
                                    if np.any(removed_area[row, x] != 0):
                                        rightmost_non_zero_x = x
                                        break
                                if leftmost_non_zero_x is not None and rightmost_non_zero_x is not None:
                                    midpoint_row = (leftmost_non_zero_x + rightmost_non_zero_x) // 2
                                    cv2.line(green_overlay, (leftmost_non_zero_x, row), (midpoint_row, row), (0, 255, 0), 2)

                        alpha = 0.2
                        cv2.addWeighted(green_overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
    
    
            else:
                # Draw a line from the leftmost non-zero pixel to the rightmost non-zero pixel of the last row
                last_row = height - 1
                leftmost_last_row = None
                rightmost_last_row = None
                for x in range(width):
                    if np.any(removed_area[last_row, x] != 0):
                        leftmost_last_row = x
                        break
                for x in range(width - 1, -1, -1):
                    if np.any(removed_area[last_row, x] != 0):
                        rightmost_last_row = x
                        break
                if leftmost_last_row is not None and rightmost_last_row is not None:
                    cv2.line(frame_resized_for_display, (leftmost_last_row, last_row), 
                            (rightmost_last_row, last_row), (0, 255, 0), 2)
                    
                    # Display a message
                    cv2.putText(frame_resized_for_display, "Line length < half image width", (10, height - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    # Calculate midpoints
                    midpoint_last_row = ((leftmost_last_row + rightmost_last_row) // 2, last_row)
                    
                    # Draw a line between the midpoints
                    if midpoint_10th_row is not None:
                        cv2.line(frame_resized_for_display, midpoint_10th_row, midpoint_last_row, (0, 255, 255), 2)

                        # Check the angle between the yellow line and the green line
                        slope_yellow = calcu_slope(midpoint_10th_row, midpoint_last_row)
                        slope_green = calcu_slope((leftmost_last_row, last_row), (rightmost_last_row, last_row))
                        
                        angle_radians = math.atan(abs((slope_yellow - slope_green) / (1 + slope_yellow * slope_green)))
                        angle_degrees = math.degrees(angle_radians)

                        # Check for intersection and draw green lines if no intersection
                        green_overlay = frame_resized_for_display.copy()
                        intersection_found = False
                        
                        # Iterate through each list in bottom_line_coordinates
                        if isinstance(bottom_line_coordinates, list):
                            for sublist in bottom_line_coordinates:
                                if isinstance(sublist, list):
                                    for (x, y) in sublist:
                                        if x == midpoint_last_row[0] and y == midpoint_last_row[1]:
                                            intersection_found = True
                                            break
                                if intersection_found:
                                    break
                        
                        if not intersection_found:
                            for row in range(top_line_row, height):
                                leftmost_non_zero_x = None
                                rightmost_non_zero_x = None
                                for x in range(width):
                                    if np.any(removed_area[row, x] != 0):
                                        leftmost_non_zero_x = x
                                        break
                                for x in range(width - 1, -1, -1):
                                    if np.any(removed_area[row, x] != 0):
                                        rightmost_non_zero_x = x
                                        break
                                if leftmost_non_zero_x is not None and rightmost_non_zero_x is not None:
                                    midpoint_row = (leftmost_non_zero_x + rightmost_non_zero_x) // 2
                                    left_to_mid_midpoint = ((leftmost_non_zero_x + midpoint_row) // 2, row)
                                    mid_to_right_midpoint = ((midpoint_row + rightmost_non_zero_x) // 2, row)
                                    cv2.line(green_overlay, left_to_mid_midpoint, mid_to_right_midpoint, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame_resized_for_display, "vehicle ahead", (10, height - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        alpha = 0.5
                        cv2.addWeighted(green_overlay, alpha, frame_resized_for_display, 1 - alpha, 0, frame_resized_for_display)
    
    
                        # if angle_degrees > 96:
                        #     for bottom_coords in bottom_line_coordinates:
                        #         for coord in bottom_coords:
                        #             if is_point_on_line_segment(midpoint_10th_row, midpoint_last_row, coord):
                        #                 cv2.putText(frame_resized_for_display, "Vehicle ahead, move left", (10, height - 50), 
                        #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        #                 if not pygame.mixer.music.get_busy():
                        #                     pygame.mixer.music.load('vehicle_ahead_move_left.mp3')
                        #                     pygame.mixer.music.play()
                        #                 break
                        #             else:
                        #                 cv2.putText(frame_resized_for_display, "keep left", (10, height - 50), 
                        #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        #                 if not pygame.mixer.music.get_busy():
                        #                     pygame.mixer.music.load('keep_left.mp3')
                        #                     pygame.mixer.music.play()
                        # elif angle_degrees < 86:
                        #     for bottom_coords in bottom_line_coordinates:
                        #         for coord in bottom_coords:
                        #             if is_point_on_line_segment(midpoint_10th_row, midpoint_last_row, coord):
                        #                 cv2.putText(frame_resized_for_display, "Vehicle ahead, move right", (10, height - 50), 
                        #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        #                 if not pygame.mixer.music.get_busy():
                        #                     pygame.mixer.music.load('vehicle_ahead_move_right.mp3')
                        #                     pygame.mixer.music.play()
                        #                 break
                        #             else:
                        #                 cv2.putText(frame_resized_for_display, "keep right", (10, height - 50), 
                        #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        #                 if not pygame.mixer.music.get_busy():
                        #                     pygame.mixer.music.load('keep_right.mp3')
                        #                     pygame.mixer.music.play()
                        # elif 86 < angle_degrees < 96:
                        #     for bottom_coords in bottom_line_coordinates:
                        #         for coord in bottom_coords:
                        #             if is_point_on_line_segment(midpoint_10th_row, midpoint_last_row, coord):
                        #                 cv2.putText(frame_resized_for_display, "Vehicle ahead, move left", (10, height - 50), 
                        #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        #                 if not pygame.mixer.music.get_busy():
                        #                     pygame.mixer.music.load('vehicle_ahead_move_left.mp3')
                        #                     pygame.mixer.music.play()
                        #                 break

    return frame_resized_for_display
for img_filename in os.listdir("Sample"):
    img_path = os.path.join("Sample", img_filename)
    original_img = load_img(img_path)
    original_img_array = img_to_array(original_img) / 255.0
    original_img_height, original_img_width = original_img_array.shape[:2]
    # Resize the image to the model's input size (still in RGB format)
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)                          
    # Make prediction
    prediction = model.predict(img_array_expanded)
    predicted_mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    # Resize the predicted mask to the original image size
    predicted_mask_resized = cv2.resize(predicted_mask, (original_img_width, original_img_height), interpolation=cv2.INTER_NEAREST)
    # Ensure predicted_mask_resized is 3-channel RGB
    predicted_mask_3ch = cv2.cvtColor(predicted_mask_resized, cv2.COLOR_GRAY2RGB)
    # Create a black image with the same size as the original image (RGB format)
    black_image = np.zeros((original_img_height, original_img_width, 3), dtype=np.uint8)
    # Convert images from float to uint8 for display
    original_img_display = (original_img_array * 255).astype(np.uint8)
    org = original_img_display.copy()
    org = cv2.cvtColor(org, cv2.COLOR_RGB2BGR)
    black_image[predicted_mask_resized == 255] = original_img_display[predicted_mask_resized == 255]  # RGB format
    original_img_display[predicted_mask_resized == 255] = [1, 224, 1]  # Color the mask area in RGB
    # Resize images for display
    original_img_display_resized = cv2.resize(original_img_display, (display_width, display_height), interpolation=cv2.INTER_AREA)
    black_image_resized = cv2.resize(black_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
    # Convert images to BGR format for OpenCV display
    original_img_display_resized_bgr = cv2.cvtColor(original_img_display_resized, cv2.COLOR_RGB2BGR)
    black_image_resized_bgr = cv2.cvtColor(black_image_resized, cv2.COLOR_RGB2BGR)
    roi_check = black_image_resized_bgr.copy()
    obj = black_image_resized_bgr.copy()
    pot = black_image_resized_bgr.copy()
    # Check if the predicted mask is empty (all zeros)
    if np.count_nonzero(predicted_mask_resized) == 0:
        text = "Not Road"
        frame_resized_for_display = cv2.resize(org, (display_width, display_height), interpolation=cv2.INTER_AREA)
        cv2.putText(frame_resized_for_display, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("not road", roi_check)
    else:
        text = "Road"
        frame_resized_for_display = cv2.resize(org, (display_width, display_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("road", obj)
        #top to bottom distance check
        rows_with_lane = np.any(predicted_mask_resized == 255, axis=1)
        topmost_row = np.argmax(rows_with_lane)
        bottommost_row = len(rows_with_lane) - np.argmax(rows_with_lane[::-1]) - 1
        distance_top_to_bottom = bottommost_row - topmost_row
        print(distance_top_to_bottom)
        if distance_top_to_bottom >= 800:
            continue
        # elif 700<= distance_top_to_bottom <= 800:
        #     continue
        else:
            left, right = roi(roi_check)
            if np.all(left == 0):
                cv2.putText(frame_resized_for_display, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_resized_for_display, "out of track, keep right", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("road", roi_check)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load('keep_right.mp3')
                    pygame.mixer.music.play()
            elif np.all(right == 0):
                cv2.putText(frame_resized_for_display, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_resized_for_display, "out of track, keep left", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("road", roi_check)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load('keep_left.mp3')
                    pygame.mixer.music.play()
            else:
                cv2.putText(frame_resized_for_display, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_resized_for_display, "on track", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("road", roi_check)
                yolov, removed_area, bottom, num1 = detect_image_with_lines(obj)
                cv2.imshow("removed", removed_area)
                cv2.imshow("obj", yolov)
                if num1 > 4:
                    cv2.putText(frame_resized_for_display, "Traffic", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    yolo_results = detect_objects(pot, yolo_model1)
                    plot_results(pot, yolo_results, yolo_model1.names, conf_threshold=0.5)
                    frame_resized_for_display = draw_left_right_dots_and_top_line(removed_area, frame_resized_for_display, bottom)
                    cv2.imshow("img", frame_resized_for_display)
                    cv2.imshow("YOLO Detection", pot)
    # Display the result and wait for a key press to proceed to the next image
    while True:
        if cv2.waitKey(1) & 0xFF == ord('n'):  # Press 'n' to proceed to the next image
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the looqqp
        break
    
cv2.destroyAllWindows()
pygame.quit()
# print("memory usagge after:",memory_usage())