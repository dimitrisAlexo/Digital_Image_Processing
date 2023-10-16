from localDescriptor import *


def descriptor_matching(points1, points2, img1, img2, percentageThreshold):
    num_points1 = len(points1)
    num_points2 = len(points2)

    # Compute the Euclidean distances between local descriptors
    distances = np.zeros((num_points1, num_points2))
    for i in range(num_points1):
        descriptor1 = my_local_descriptor(img1, [points1[i][1], points1[i][0]], 5, 20, 0.5, 8)
        if not descriptor1:
            continue  # Skip if descriptor1 is empty
        for j in range(num_points2):
            descriptor2 = my_local_descriptor(img2, [points2[j][1], points2[j][0]], 5, 20, 0.5, 8)
            if not descriptor2:
                continue  # Skip if descriptor2 is empty
            distance = np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))
            distances[i, j] = distance

    # Compute the threshold to select a percentage of point pairs
    threshold = np.percentile(distances, percentageThreshold)

    # Find the matched point pairs based on the threshold
    matching_points = []
    for i in range(num_points1):
        for j in range(num_points2):
            if distances[i, j] <= threshold:
                matching_points.append((points1[i], points2[j]))

    return matching_points
