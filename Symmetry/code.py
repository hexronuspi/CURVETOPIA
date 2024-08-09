import sys
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

from scipy.spatial import KDTree

from PIL import Image
import seaborn as sns
import glob

class BilateralDetector:
    def __init__(self):
        self.image = None
        self.reflected_image = None
        self.keypoints = None
        self.reflected_keypoints = None
        self.matchpoints = None
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.symmetry_line = (None, None)
        self.points_r = None
        self.points_theta = None

    def find(self, source_path):
        self.image = read_bgr_image(source_path)
        self.reflected_image = np.fliplr(self.image)

        self.keypoints, descriptors = self.sift.detectAndCompute(self.image, None)
        self.reflected_keypoints, reflected_descriptors = self.sift.detectAndCompute(self.reflected_image, None)

        self.matchpoints = self.match_descriptors(descriptors, reflected_descriptors)
        matchpoints_weight = [self.calculate_symmetry_match(match) for match in self.matchpoints]
        potential_symmetry_axis = [self.get_potential_symmetry_axis(match) for match in self.matchpoints]
        self.points_r = [rij for rij, xc, yc, theta in potential_symmetry_axis]
        self.points_theta = [theta for rij, xc, yc, theta in potential_symmetry_axis]

        image_hexbin = plt.hexbin(self.points_r, self.points_theta, bins=50, cmap=plt.cm.Spectral_r)
        sorted_vote = sort_hexbin_by_votes(image_hexbin)
        r, theta = find_coordinate_maxhexbin(sorted_vote)
        self.symmetry_line = (r, theta)
        plt.close()
        return r, theta

    def get_potential_symmetry_axis(self, match):
        pi = self.keypoints[match.queryIdx]
        pj = self.reflected_keypoints[match.trainIdx]
        normalize_angle(pi)
        normalize_angle(pj)
        pj.pt = (self.image.shape[1] - pj.pt[0], pj.pt[1])
        theta = angle_with_x_axis(pi.pt, pj.pt)
        xc, yc = midpoint(pi, pj)
        rij = xc * np.cos(theta) + yc * np.sin(theta)
        return rij, xc, yc, theta

    def match_descriptors(self, descriptors, reflected_descriptors):
        matchpoints = [item[0] for item in self.bf.knnMatch(descriptors, reflected_descriptors, k=2)]
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)
        return matchpoints

    def calculate_symmetry_match(self, match):
        pi = self.keypoints[match.queryIdx]
        pj = self.reflected_keypoints[match.trainIdx]
        normalize_angle(pi)
        normalize_angle(pj)
        pj.pt = (self.image.shape[1] - pj.pt[0], pj.pt[1])
        theta = angle_with_x_axis(pi.pt, pj.pt)
        angular_symmetry = 1 - np.cos(pj.angle + pi.angle - 2 * theta)
        scale_symmetry = np.exp((-abs(pi.size - pj.size) / ((pi.size + pj.size)))) ** 2
        d = (pj.pt[0] - pi.pt[0]) ** 2 + (pj.pt[1] - pi.pt[1]) ** 2
        distance_weight = np.exp(-d ** 2 / 2)
        return angular_symmetry * scale_symmetry * distance_weight

class SymmetryDrawer:
    def __init__(self):
        pass

    def draw_symmetry(self, bilateral_detector):
        r, theta = bilateral_detector.symmetry_line
        image = bilateral_detector.image.copy()
        height, width, _ = image.shape
        for y in range(height):
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < width:
                    image[y, x] = [0, 0, 0]
                    image[y, x + 1] = [0, 0, 0]
                    image[y, x - 1] = [0, 0, 0]
            except IndexError:
                continue
        return image

def read_bgr_image(image_path):
    image = cv2.imread(image_path)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return im_rgb

def normalize_angle(point):
    point.angle = np.deg2rad(point.angle)
    point.angle = np.pi - point.angle
    if point.angle < 0.0:
        point.angle += 2 * np.pi

def midpoint(pi, pj):
    return pi.pt[0] / 2 + pj.pt[0] / 2, pi.pt[1] / 2 + pj.pt[1] / 2

def angle_with_x_axis(pi, pj):
    x, y = pi[0] - pj[0], pi[1] - pj[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

def sort_hexbin_by_votes(image_hexbin):
    counts = image_hexbin.get_array()
    ncnts = np.count_nonzero(np.power(10, counts))
    verts = image_hexbin.get_offsets()
    output = {}
    for offc in range(verts.shape[0]):
        binx, biny = verts[offc][0], verts[offc][1]
        if counts[offc]:
            output[(binx, biny)] = counts[offc]
    return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

def find_coordinate_maxhexbin(sorted_vote):
    for k, v in sorted_vote.items():
        if k[1] == 0 or k[1] == np.pi:
            continue
        else:
            return k[0], k[1]
