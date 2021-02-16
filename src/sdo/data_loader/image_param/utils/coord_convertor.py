# Credit A.Ahmadzadeh, https://bitbucket.org/gsudmlab/imageparams_api/src/master/
from shapely.geometry import Polygon


def convert_boundingpoints_to_pixelunit(polygon: Polygon, cdelt1, cdelt2, crpix1, crpix2, original_w, shrinkage_ratio=1):
    """
    This method converts the points coordinates from arc-sec unit to pixel unit, and meanwhile
    makes 2 modifications:
        1. Vertical mirror of the points (this is required if JPG format is being used)
        2. Shrinkage of points (required if downsized images are being used.)

    :param polygon: a list of points forming a closed shape.
    :param cdelt1: fits/jp2 header information to scale in x direction
    :param cdelt2: fits/jp2 header information to scale in y direction
    :param crpix1: fits/jp2 header information to shift in x direction
    :param crpix2: fits/jp2 header information to shift in y direction
    :param original_w: the width of the original images. It is assumed that the images
    are in square shape, hence width and height are equal.
    :param shrinkage_ratio: a float point that indicates the ratio (original_w/new_size).
    For example, for 512 X 512 image, it should be 4.0.
    :return: a polygon object (from Shapely package) and None if the list was empty. If you need
            a list of tuples (x, y) instead, you can convert it using `poly = poly.exterior.coords'
    """
    points = polygon.exterior.coords
    b = [(float(v[0]) / cdelt1 + crpix1, float(v[1]) / cdelt2 + crpix2)
         for v in points]

    # Shrink and then mirror vertically
    b = [(v[0] / shrinkage_ratio, (original_w - v[1]) / shrinkage_ratio)
         for v in b]

    return Polygon(b)
