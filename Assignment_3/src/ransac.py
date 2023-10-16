import numpy as np
import random


def calculate_h(pair1, pair2):
    # Extract points from the pairs
    point1_1, point1_2 = pair1
    point2_1, point2_2 = pair2

    # Calculate vectors from the points
    vector1 = point2_1 - point1_1
    vector2 = point2_2 - point1_2

    # Calculate the angle theta
    theta = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    d = (point1_2.T - np.dot(R, point1_1.T)).T

    # Create the dictionary H
    H = {'theta': theta, 'd': d}

    return H


def transform_points(points, H):
    theta = H['theta']  # in rad
    d = H['d']

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Define the transformation matrix R
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    transformed_points = np.dot(R, np.array(points).T).T + d

    return transformed_points.astype(int)


def my_RANSAC(matching_points, r, N):
    H = {'theta': 0, 'd': [0, 0]}
    inlier_matching_points = []
    outlier_matching_points = []
    best_distance = []

    score = 0

    for i in range(N):
        pair1, pair2 = random.sample(matching_points, 2)

        H_temp = calculate_h(pair1, pair2)

        points1 = [pair[0] for pair in matching_points]
        points2 = [pair[1] for pair in matching_points]
        transformed_points1 = transform_points(points1, H_temp)
        transformed_points2 = transform_points(points2, {'theta': 0, 'd': [0, 0]})

        distance = np.linalg.norm(transformed_points2 - transformed_points1, axis=1)
        count = np.count_nonzero(distance < 80)

        if count > score:
            score = count
            best_distance = distance
            H['theta'] = H_temp['theta']
            H['d'] = H_temp['d']

    for i in range(len(best_distance)):
        if best_distance[i] < r:
            inlier_matching_points.append(matching_points[i])
        else:
            outlier_matching_points.append(matching_points[i])

    return H, inlier_matching_points, outlier_matching_points
