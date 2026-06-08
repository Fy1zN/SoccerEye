import math


def get_center_of_bbox(bbox):

    x1, y1, x2, y2 = bbox

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )

    return center_x, center_y


def get_bbox_width(bbox):

    return bbox[2] - bbox[0]


def get_foot_position(bbox):

    x1, y1, x2, y2 = bbox

    x = int(
        (x1 + x2) / 2
    )

    y = int(y2)

    return x, y


def measure_distance(
    p1,
    p2
):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def measure_xy_distance(
    p1,
    p2
):

    return (
        p1[0] - p2[0],
        p1[1] - p2[1]
    )