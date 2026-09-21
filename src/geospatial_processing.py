import numpy as np


def euclidean_distance(point_a, point_b):
    """
    Compute the Euclidean distance between two geographical points.
    """
    point_a = np.asarray(point_a)
    point_b = np.asarray(point_b)

    return np.linalg.norm(point_a - point_b)


def project_point_on_segment(point, start, end):
    """
    Compute the orthogonal projection of a point onto a segment.

    Parameters
    ----------
    point : array-like
        Point to project.

    start : array-like
        First endpoint of the segment.

    end : array-like
        Second endpoint of the segment.

    Returns
    -------
    numpy.ndarray
        Coordinates of the projected point.
    """

    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    segment = end - start

    denominator = np.dot(segment, segment)

    if denominator == 0:
        return start

    projection_factor = ( np.dot(point - start, segment) / denominator  )

    # Restrict projection to the segment
    projection_factor = np.clip( projection_factor, 0.0,  1.0 )

    return start + projection_factor * segment


def closest_point_on_trajectory(point, trajectory):
    """
    Find the closest projected point on a trajectory.

    The trajectory is represented by a sequence of points.
    """

    if len(trajectory) < 2:
        raise ValueError("A trajectory must contain at least two points."  )

    best_projection = None
    minimum_distance = float("inf")

    for i in range(len(trajectory) - 1):

        start = trajectory[i]
        end = trajectory[i + 1]

        projection = project_point_on_segment( point, start, end  )
        distance = euclidean_distance( point, projection )

        if distance < minimum_distance:
            minimum_distance = distance
            best_projection = projection

    return best_projection, minimum_distance
