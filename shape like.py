import numpy as np
from scipy.spatial import ConvexHull

def sphericity(points):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Calculate the volume of the convex hull
    hull_volume = hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = hull.area

    # Calculate the radius of a sphere with the same volume
    sphere_radius = ((3 * hull_volume) / (4 * np.pi))**(1/3)

    # Calculate the surface area of a sphere with the same volume
    sphere_surface_area = 4 * np.pi * (sphere_radius**2)

    # Calculate the sphericity
    sphericity = hull_surface_area / sphere_surface_area

    return sphericity

def cubic_sphericity(points):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Calculate the volume of the convex hull
    hull_volume = hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = hull.area

    # Calculate the side length of a cube with the same volume
    cube_side_length = (hull_volume)**(1/3)

    # Calculate the surface area of a cube with the same volume
    cube_surface_area = 6 * (cube_side_length**2)

    # Calculate the cubic sphericity
    cubic_sphericity = hull_surface_area / cube_surface_area

    return cubic_sphericity


def cone_sphericity(points):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Calculate the volume of the convex hull
    hull_volume = hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = hull.area

    # Calculate the radius and height of a cone with the same volume
    cone_radius = np.sqrt(hull_surface_area / (np.pi * hull_volume))
    cone_height = hull_volume / (np.pi * cone_radius**2)

    # Calculate the surface area of a cone with the same volume
    cone_surface_area = np.pi * cone_radius * (cone_radius + np.sqrt(cone_radius**2 + cone_height**2))

    # Calculate the cone sphericity
    cone_sphericity = hull_surface_area / cone_surface_area

    return cone_sphericity
