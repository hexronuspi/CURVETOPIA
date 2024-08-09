import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    return val

def convexHull(points):
    n = len(points)
    if n < 3:
        return []
    
    l = 0
    for i in range(1, len(points)):
        if points[i][0] < points[l][0]:
            l = i
        elif points[i][0] == points[l][0]:
            if points[i][1] > points[l][1]:
                l = i
    
    hull = []
    p = l
    while True:
        hull.append(points[p])
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) < 0:
                q = i
        p = q
        if p == l:
            break

    return hull

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def cost(p1, p2, p3):
    return dist(p1, p2) + dist(p2, p3) + dist(p3, p1)

def triangulation(points):
    n = len(points)
    if n < 3:
        return []
    
    table = [[(float('inf'), -1)] * n for _ in range(n)]
    for gap in range(n):
        for j in range(gap, n):
            i = j - gap
            if j < i + 2:
                table[i][j] = (0.0, -1)
            else:
                for k in range(i + 1, j):
                    val = table[i][k][0] + table[k][j][0] + cost(points[i], points[j], points[k])
                    if table[i][j][0] > val:
                        table[i][j] = (val, k)
    
    queue = [[0, n - 1]]
    triangles = []
    while queue:
        x = queue.pop(0)
        k = table[x[0]][x[1]][1]
        if k - x[0] > 2:
            queue.append([x[0], k])
        if x[1] - k > 2:
            queue.append([k, x[1]])
        triangles.append([points[x[0]], points[k], points[x[1]]])
    
    return triangles

def detect_shapes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 3:
            if len(approx) == 3:
                shape = 'Triangle'
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                shape = 'Rectangle' if 0.9 <= aspect_ratio <= 1.1 else 'Quadrilateral'
            else:
                area = cv2.contourArea(approx)
                hull = cv2.convexHull(approx, returnPoints=True)
                hull_area = cv2.contourArea(hull)
                shape = 'Circle' if abs(1 - (area / hull_area)) <= 0.2 else 'Polygon'
            shapes.append((shape, approx))
    
    return shapes

def plot_shapes_and_triangles(image_path):
    shapes = detect_shapes(image_path)
    
    image = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    
    for shape, contour in shapes:
        polygon = [(point[0][0], point[0][1]) for point in contour]
        convex_polygon = convexHull(polygon)
        polygon_patch = patches.Polygon(convex_polygon, closed=True, edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(polygon_patch)
        
        color = {'Triangle': 'blue', 'Rectangle': 'green', 'Circle': 'orange', 'Polygon': 'purple'}
        ax.text(np.mean([p[0] for p in polygon]), np.mean([p[1] for p in polygon]),
                shape, color=color.get(shape, 'black'), fontsize=12, weight='bold')

        triangles = triangulation(convex_polygon)
        for triangle in triangles:
            triangle_patch = patches.Polygon(triangle, closed=True, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(triangle_patch)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

image_path = 'a6.png'
plot_shapes_and_triangles(image_path)