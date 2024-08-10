# Adobe GenSolve 2024 Round 2

This repository contains the codebase for **Adobe GenSolve 2024 Round 2**.

<hr>
This codebase can be visualised in colab : 

1. <a href = 'https://colab.research.google.com/drive/1dtSS-wcZi-1UJ5yARUKFxxfauVi0B3hm?usp=sharing'> Colab </a>

2. Under `Adobe_GenSolve_Round_2.ipynb`

<hr>
**Team:**
- **Sakshi Kumari**
- **Aditya Raj**  (Team Lead)

Email (Team Lead): 
adityar.ug22.ec@nitp.ac.in

<hr>
**Run** **bold text**

Just select `Runtime` and select `run all`.
<hr>

## Directory Structure 

The project is organized into the following directories:
- **Regularisation**
- **Symmetry**
- **Detection/**
  - **Genetics**
  - **Pure Mathematics**
  - **Triangulation Method**
- **Shape Completion** 


### 1. Shape Regularization

Designed and implemented a Shape Regularization Pipeline in Python. Developed algorithms to classify and regularize geometric shapes polygons, rectangles, squares, circles, ellipses, and triangles, ensuring the shapes conform to predefined criteria. Employed advanced contour analysis techniques for shape detection and regularization, with visual outputs generated for original and regularized images. 


### 2. Symmetry Detection

Developed a Python-based Bilateral Symmetry Detection System. Implemented keypoint detection using SIFT and descriptor matching with a Brute Force matcher to identify and visualize the symmetry axis in images. Processed and refined keypoint data through hexbin analysis and advanced mathematical transformations to calculate potential symmetry axes, resulting in an accurate symmetry line detection and visualization. 

### 3. Detection Techniques

- **Genetics:**
  This approach combines a Genetic Algorithm (GA) with Neural Networks (NN) to detect 2D shapes in images, A hybrid Genetic Algorithm-Neural Network model for shape classification, utilizing custom activation functions and gradient-based optimization. Implemented mutation, crossover, and selection techniques in a population-based algorithm, it did not achieve significant reduction across epochs.

- **Pure Mathematics:**
  Detect and classify various shapes in an image, such as circles, ellipses, rectangles, squares, regular polygons, rounded rectangles, stars, and straight lines. The process involves contour detection, shape classification based on geometric properties. **It works with overlapping image**.

- **Triangulation Method:**
  A shape detection and triangulation system using Python. The system accurately identifies and classifies geometric shapes such as triangles, rectangles, circles, ellipse, rectangle, polygons, and unknown shapes by analyzing contour properties like aspect ratios, convex hulls, and contour areas. Implemented custom algorithms to detect convex hulls and perform shape triangulation, enabling the precise visualization of detected shapes with annotated labels and triangulated segments. The edge detection and contour approximation techniques were optimized for high performance, allowing the system to handle large images efficiently, but it does not work with overlapping images.

### 4. Shape Completion

Utilized numpy for data manipulation and scipy for spatial computations, including ConvexHull and Delaunay triangulation to complete shapes. Implemented Bézier curve fitting using sklearn's PolynomialFeatures and LinearRegression to generate smooth curves from point data. The script integrates Bézier curves, and completes shapes, providing shape completion.






