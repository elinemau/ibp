import numpy as np
from scipy.spatial import ConvexHull

def rotating_calipers(points):
    hull = ConvexHull(points)
    convex_hull_points = points[hull.vertices]  # Extract points on the convex hull

    min_width = float('inf')
    min_width_points = None

    for i in range(len(convex_hull_points)):
        p1 = convex_hull_points[i]
        p2 = convex_hull_points[(i + 1) % len(convex_hull_points)]

        # Compute the direction vector of the current edge
        edge_vector = p2 - p1

        # Rotate the convex hull points to align with the current edge
        rotated_points = convex_hull_points.dot(np.array([[edge_vector[0], edge_vector[1]], [-edge_vector[1], edge_vector[0]]]))

        # Compute the minimum and maximum x-coordinates of the rotated points
        min_x = np.min(rotated_points[:, 0])
        max_x = np.max(rotated_points[:, 0])

        # Compute the width of the bounding box
        width = max_x - min_x

        # Update the minimum width if necessary
        if width < min_width:
            min_width = width
            min_width_points = rotated_points

    return min_width, min_width_points

# Example usage
np.random.seed(42)
points = np.random.rand(30, 2)  # Replace this with your set of 2D points

min_width, min_width_points = rotating_calipers(points)

print("Minimum Width:", min_width)
print("Points of Minimum Width Rectangle:")
print(min_width_points)
