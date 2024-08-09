from shapely.geometry import Polygon, Point, LineString
import matplotlib.patches as patches
from shapely.affinity import scale
from shapely.geometry.polygon import orient

import bezier
from bezier import Curve

class ShapeRegularizer:
    def __init__(self, num_points=100, alpha=0.1):
        self.num_points = num_points
        self.alpha = alpha

    def regularize_polygon(self, polygon):
        coords = np.array(polygon.exterior.coords)
        if len(coords) < 4:
            return polygon

        new_coords = []
        for i in range(len(coords)):
            prev_idx = (i - 1) % len(coords)
            next_idx = (i + 1) % len(coords)
            new_x = (1 - self.alpha) * coords[i][0] + self.alpha * (coords[prev_idx][0] + coords[next_idx][0]) / 2
            new_y = (1 - self.alpha) * coords[i][1] + self.alpha * (coords[prev_idx][1] + coords[next_idx][1]) / 2
            new_coords.append((new_x, new_y))
        return Polygon(new_coords)

    def regularize_rectangle(self, rectangle):
        coords = np.array(rectangle.exterior.coords)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        rect_coords = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
        return Polygon(rect_coords)

    def regularize_square(self, square):
        coords = np.array(square.exterior.coords)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        side_length = min(max_x - min_x, max_y - min_y)
        new_coords = [(min_x, min_y), (min_x + side_length, min_y),
                      (min_x + side_length, min_y + side_length), (min_x, min_y + side_length), (min_x, min_y)]
        return Polygon(new_coords)

    def regularize_circle(self, circle):
        center = circle.centroid
        coords = np.array(circle.exterior.coords)
        distances = np.linalg.norm(coords - np.array([center.x, center.y]), axis=1)
        radius = np.mean(distances)
        angles = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        circle_coords = [(center.x + radius * np.cos(angle), center.y + radius * np.sin(angle)) for angle in angles]
        circle_coords.append(circle_coords[0])
        return Polygon(circle_coords)


    def regularize_ellipse(self, ellipse):
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(np.array(ellipse.exterior.coords))
            major_axis = MA / 2
            minor_axis = ma / 2
            center = Point(x, y)
            angles = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
            ellipse_coords = [(center.x + major_axis * np.cos(angle), center.y + minor_axis * np.sin(angle)) for angle in angles]
            return Polygon(ellipse_coords)
        except cv2.error:
            return ellipse

    def regularize_triangle(self, triangle):
        coords = np.array(triangle.exterior.coords)
        if len(coords) != 4:
            return triangle

        centroid = np.mean(coords[:-1], axis=0)

        side_lengths = [np.linalg.norm(coords[i] - coords[(i+1) % 3]) for i in range(3)]
        max_side = max(side_lengths)
        new_coords = []
        for i in range(3):
            x = centroid[0] + (coords[i][0] - centroid[0]) * max_side / side_lengths[i]
            y = centroid[1] + (coords[i][1] - centroid[1]) * max_side / side_lengths[i]
            new_coords.append((x, y))
        new_coords.append(new_coords[0])
        return Polygon(new_coords)

    def classify_shape(self, contour):
        contour = np.squeeze(contour)
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)

        if contour.shape[0] < 3:
            return 'unknown'

        poly = Polygon(contour)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(np.array(contour), epsilon, True)
        num_vertices = len(approx)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 'unknown'

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.8 and self.is_nearly_circular(contour):
            return 'circle'

        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            if MA / ma > 1.5:
                return 'ellipse'
        except cv2.error:
            pass

        if num_vertices == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.95 < aspect_ratio < 1.05:
                return 'square'
            else:
                return 'rectangle'

        if num_vertices == 3:
            if self.is_triangle_by_angle(contour):
                return 'triangle'

        if num_vertices > 5:
            if self.is_star_shape(contour):
                return 'star'
            else:
                return 'polygon'

        return 'unknown'

    def is_nearly_circular(self, contour):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(np.array(contour), epsilon, True)
        num_vertices = len(approx)

        if num_vertices > 4:
            return True

        return False

    def is_triangle_by_angle(self, contour):
        contour = np.array(contour)
        angles = []
        for i in range(3):
            pt1, pt2, pt3 = contour[i % 3], contour[(i + 1) % 3], contour[(i + 2) % 3]
            vec1 = pt2 - pt1
            vec2 = pt3 - pt2
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angles.append(np.degrees(angle))

        if np.isclose(sum(angles), 180, atol=5) and all(50 <= angle <= 80 for angle in angles):
            return True
        return False

    def is_star_shape(self, contour):
        contour = np.array(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return False

        num_vertices = len(contour)

        if num_vertices < 5:
            return False

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area / area < 2:
            return True

        return False

    def regularize_shape(self, contour):
        contour = np.squeeze(contour)
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            return Polygon(contour)
        poly = Polygon(contour)
        shape_type = self.classify_shape(contour)
        if shape_type == 'rectangle':
            return self.regularize_rectangle(poly)
        elif shape_type == 'square':
            return self.regularize_square(poly)
        elif shape_type == 'circle':
            return self.regularize_circle(poly)
        elif shape_type == 'ellipse':
            return self.regularize_ellipse(poly)
        elif shape_type == 'triangle':
            return self.regularize_triangle(poly)
        elif shape_type == 'polygon':
            return self.regularize_polygon(poly)
        else:
            return poly

    def draw_regularized_shapes(self, image, contours):
        output_image = np.zeros_like(image)
        for contour in contours:
            regularized_shape = self.regularize_shape(contour)
            if len(regularized_shape.exterior.coords) < 4:
                continue
            regularized_coords = np.array(regularized_shape.exterior.coords, dtype=np.int32)
            cv2.polylines(output_image, [regularized_coords], isClosed=True, color=(255, 255, 255), thickness=2)
        return output_image.astype(np.uint8)

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path, invert=True):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
        if binary is None:
            raise ValueError("Thresholding failed.")
        return binary.astype(np.uint8)

    @staticmethod
    def extract_contours(binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class ShapeRegularizationPipeline:
    def __init__(self, num_points=100, alpha=0.1):
        self.regularizer = ShapeRegularizer(num_points=num_points, alpha=alpha)

    def regularize(self, input_path, output_path):
        binary_image = ImageProcessor.preprocess_image(input_path, invert=True)
        contours = ImageProcessor.extract_contours(binary_image)
        output_image = self.regularizer.draw_regularized_shapes(binary_image, contours)
        cv2.imwrite(output_path, output_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(binary_image, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Regularized Shapes")
        plt.imshow(output_image, cmap='gray')

        plt.show()

pipeline = ShapeRegularizationPipeline(num_points=1000, alpha=0.9)
pipeline.regularize(f'/content/doodle.png', '/content/output_regularised1.png')