import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.polygon import orient
from math import sqrt, pi

def detect_shapes(input_path):
    def classify_shape(contour):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices < 3:
            return 'unknown'
        
        if num_vertices >= 3:
            points = [tuple(pt[0]) for pt in approx]
            if len(points) < 3:
                return 'unknown'
            try:
                poly = Polygon(points)
                area = poly.area
                perimeter = poly.length

                if num_vertices > 5:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if circularity > 0.8:
                        return 'circle'
                    else:
                        return 'ellipse'
                
                if num_vertices == 4:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 0.9 <= aspect_ratio <= 1.1:
                        return 'square'
                    else:
                        return 'rectangle'

                if num_vertices >= 5:
                    angles = []
                    for i in range(num_vertices):
                        p1 = np.array(approx[i][0])
                        p2 = np.array(approx[(i + 1) % num_vertices][0])
                        p3 = np.array(approx[(i + 2) % num_vertices][0])
                        angle = np.arccos(np.clip(np.dot(p2 - p1, p3 - p2) / (np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p2)), -1.0, 1.0))
                        angles.append(np.degrees(angle))
                    angles = np.array(angles)
                    if np.all(np.abs(angles - angles[0]) < 10):
                        return 'regularPolygon'

                if num_vertices == 4:
                    contours_approx = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
                    for approx in contours_approx:
                        rect = cv2.boundingRect(approx)
                        x, y, w, h = rect
                        if cv2.contourArea(approx) > 0.8 * (w * h):
                            return 'roundedRectangle'
                
                if num_vertices >= 5:
                    distances = [np.linalg.norm(np.array(approx[i][0]) - np.array(approx[(i + 1) % num_vertices][0])) for i in range(num_vertices)]
                    if np.all(np.abs(np.diff(distances)) < 10):
                        return 'star'

                if num_vertices == 2:
                    return 'straightLine'
            
            except ValueError:
                return 'unknown'

        return 'unknown'

    def draw_and_count_shapes(image, contours):
        shape_counts = {name: 0 for name in class_names}
        
        for contour in contours:
            shape = classify_shape(contour)
            if shape in shape_counts:
                shape_counts[shape] += 1
            else:
                shape_counts['unknown'] += 1
            
            (x, y, w, h) = cv2.boundingRect(contour)
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        return shape_counts

    class_names = ['circle', 'ellipse', 'rectangle', 'regularPolygon', 'roundedRectangle', 'star', 'straightLine', 'unknown']
    
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_counts = draw_and_count_shapes(image, contours)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    print("Detected shapes and their counts:")
    for shape, count in shape_counts.items():
        print(f"{shape}({count})")

    return shape_counts

shape_counts = detect_shapes('output_regularised1.png')