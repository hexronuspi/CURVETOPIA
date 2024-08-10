import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.spatial import ConvexHull, Delaunay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def bezier_curve(points, n=3, num_points=1000):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(n + 1):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += np.outer(bernstein, points[i])
    return curve

def fit_bezier(XYs, n=3):
    bezier_curves = []
    for XY in XYs:
        if len(XY) > n:
            X = np.linspace(0, 1, len(XY)).reshape(-1, 1)
            poly = PolynomialFeatures(degree=n).fit_transform(X)
            model = LinearRegression().fit(poly, XY)
            t = np.linspace(0, 1, 100).reshape(-1, 1)
            poly_t = PolynomialFeatures(degree=n).fit_transform(t)
            bezier_points = model.predict(poly_t)
            bezier_curves.append(bezier_points)
    return bezier_curves

def complete_shapes(XYs):
    completed_shapes = []
    for XY in XYs:
        if len(XY) > 5:
            try:
                hull = ConvexHull(XY)
                completed_shapes.append(XY[hull.vertices])
            except:
                if len(XY) > 3:
                    delaunay = Delaunay(XY)
                    for simplex in delaunay.simplices:
                        completed_shapes.append(XY[simplex])
    return completed_shapes

def plot(path_XYs, bezier_curves, completed_shapes, colours):
    fig, ax = plt.subplots(tight_layout=True, figsize=(12, 8))
    
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, markersize=5)
  
    for i, bezier in enumerate(bezier_curves):
        c = colours[i % len(colours)]
        if bezier.size > 0: 
            ax.plot(bezier[:, 0], bezier[:, 1], c=c, linewidth=2)

    for i, shape in enumerate(completed_shapes):
        c = colours[i % len(colours)]
        ax.plot(shape[:, 0], shape[:, 1], c=c, linewidth=2)

    ax.set_aspect('equal')
    ax.set_title('Bézier Curves and Completed Shapes')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.grid(True)
    plt.show()

csv_path = '/content/problems/problems/occlusion1.csv'

path_XYs = read_csv(csv_path)
bezier_curves = fit_bezier(path_XYs, n=3)
completed_shapes = complete_shapes([XY for XYs in path_XYs for XY in XYs])
plot(path_XYs, bezier_curves, completed_shapes, colours=['r', 'b', 'g', 'y'])
