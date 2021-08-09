# Credit A.Ahmadzadeh, https://bitbucket.org/gsudmlab/imageparams_api/src/master/
from PIL import Image
import numpy as np
import requests
from xml.etree import ElementTree as et
from io import BytesIO
from datetime import datetime
from sdo.data_loader.image_param.IP_CONSTANTS import URL_STRINGS
from sdo.data_loader.image_param.IP_CONSTANTS.CONSTANTS import *

'''
This module is the core of this project, which contains all necessary functions
to get data from our Image Parameter dataset through the web API, http://dmlab.cs.gsu.edu/dmlabapi/.
'''


def get_aia_image_jpeg(starttime: datetime,
                       aia_wave: (AIA_WAVE, str),
                       image_size: (IMAGE_SIZE, str)) -> Image:
    """
    queries the AIA image corresponding to the given start time, wave channel, and size.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
    to provide a valid wavelength, or pass a string from this list:
    ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']
    :param image_size: size of the image. Either use the class `constants.IMAGE_SIZE` to provide
    a valid size, or path a string from the list: ['2k', '512', '256']
    :return: the AIA image as a PIL.Image object
    """
    prepared_url = prepare_url_get_aia_image_jpeg(
        starttime, aia_wave, image_size)
    response = requests.get(prepared_url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img


def get_aia_imageparam_jpeg(starttime: datetime,
                            aia_wave: (AIA_WAVE, str),
                            image_size: (IMAGE_SIZE, str),
                            param_id: (IMAGE_PARAM, str)) -> Image:
    """
    queries the heatmap of the given image parameter when applied on the AIA image corresponding
    to the given start time, wave channel, and size.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
    to provide a valid wavelength, or pass in a string from this list:
    ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']
    :param image_size: size of the image. Either use the class `constants.IMAGE_SIZE` to provide
    a valid size, or pass in a string from the list: ['2k', '512', '256']
    :param param_id: id of the image parameters. Either use the class `constants.IMAGE_PARAM` to
    provide a valid id, or pass in a string from the list: ['1', '2', '3', '4', '5', '6', '7', '8',
    '9', '10]. To know which item corresponds to which parameter, see [dmlab.cs.gsu.aedu/dmlabapi/]
    :return: the heatmap of the given image parameter as a PIL.Image object.
    """
    prepared_url = prepare_url_get_aia_imageparam_jpeg(
        starttime, aia_wave, image_size, param_id)
    response = requests.get(prepared_url, timeout=30)
    img = Image.open(BytesIO(response.content))
    return img


def get_aia_imageparam_xml(starttime: datetime, aia_wave: (AIA_WAVE, str)) -> et:
    """
    queries the XML of 10 image parameters computed on the image corresponding
    to the given date and wavelength channel.

    Note: Use `convert_param_xml_to_ndarray` to convert the retrieved XML into a `numpy.ndarray`
    object.

    :param starttime: datetime corresponding to the requested image.
    :param aia_wave: wavelength channel corresponding to the requested image.
    :return: an `xml.etree.ElementTree.Element` instance, as the content of the
    retrieved XML.
    """
    prepared_url = prepare_url_get_aia_imageparam_xml(starttime, aia_wave)
    response = requests.get(prepared_url, timeout=30)
    xml_content = et.fromstring(response.content)
    return xml_content


def get_aia_imageparam_bulk_xml(starttime: datetime,
                                endtime: datetime,
                                aia_wave: (AIA_WAVE, str),
                                limit=100,
                                offset=0,
                                step=1) -> et:
    prepared_url = prepare_url_get_aia_imageparam_bulk_xml(
        starttime, endtime, aia_wave, limit, offset, step)
    response = requests.get(prepared_url, timeout=30)
    xml_content = et.fromstring(response.content)
    return xml_content


def get_aia_imageparam_json(starttime: datetime, aia_wave: (AIA_WAVE, str)) -> np.ndarray:
    """
    queries the JSON of 10 image parameters computed on the image corresponding
    to the given date and wavelength channel. The received JSON content will first
    be converted to an `np.ndarray` and then returned.

    Note: Use `convert_param_json_to_ndarray` to convert the retrieved XML into a `numpy.ndarray`
    object.

    :param starttime: datetime corresponding to the requested image.
    :param aia_wave: wavelength channel corresponding to the requested image.
    :return: a numpy array of the requested image parameter values.
    """
    prepared_url = prepare_url_get_aia_imageparam_json(starttime, aia_wave)
    response = requests.get(prepared_url, timeout=30)
    return response.json()


def get_aia_imageheader_xml(starttime: datetime, aia_wave: (AIA_WAVE, str)) -> et:
    """
    queries the XML of the header information of the image corresponding to the
    given date and wavelength channel.

    Note: Use `convert_header_xml_to_dict` to convert the retrieved XML into a `dict`
    object.

    The documentation for the values is as follows:
     - X0: X Coordinate of the center of Sun's disk in 4096X4096-pixel images, in pixels. [Integer]
     - Y0: Y Coordinate of the center of Sun's disk in 4096X4096-pixel images, in pixels. [Integer]
     - DSUN: Distance from Sun's center to SDO in meter. [Double]
     - R_SUN: Radius of Sun in the image. [Double]
     - CDELT: Pixel spacing per index value along image axes.
     - QUALITY: A 32-bit binary number, whose each bit represents one quality issue.

    :param starttime: datetime corresponding to the requested image header.
    :param aia_wave: wavelength channel corresponding to the requested image header.
    :return: a `dict` instance, as the content of the retrieved header.
    """
    prepared_url = prepare_url_get_aia_imageheader_xml(starttime, aia_wave)
    response = requests.get(prepared_url, timeout=30)
    xml_content = et.fromstring(response.content)
    return xml_content


def get_aia_imageheader_json(starttime: datetime, aia_wave: (AIA_WAVE, str)) -> dict:
    """
    queries the JSON of the header information of the image corresponding to the
    given date and wavelength channel.

    The documentation for the values is as follows:
     - X0: X Coordinate of the center of Sun's disk in 4096X4096-pixel images, in pixels. [Integer]
     - Y0: Y Coordinate of the center of Sun's disk in 4096X4096-pixel images, in pixels. [Integer]
     - DSUN: Distance from Sun's center to SDO in meter. [Double]
     - R_SUN: Radius of Sun in the image. [Double]
     - CDELT: Pixel spacing per index value along image axes [Float].
     - QUALITY: A 32-bit binary number, whose each bit represents one quality issue.

    :param starttime: datetime corresponding to the requested image header.
    :param aia_wave: wavelength channel corresponding to the requested image header.
    :return: a `dict` instance, as the content of the retrieved header.
    """
    prepared_url = prepare_url_get_aia_imageheader_json(starttime, aia_wave)
    response = requests.get(prepared_url, timeout=30)
    return response.json()


def get_aia_spatiotemporal_json(starttime: datetime,
                                endtime: datetime,
                                table_name: TABLE_NAME,
                                predicate: PREDICATE,
                                sort_by,
                                limit,
                                offset) -> list:
    prepared_url = prepare_url_get_aia_spatiotemporal_json(starttime, endtime, table_name,
                                                           predicate, sort_by, limit, offset)
    response = requests.get(prepared_url, timeout=30)
    return list(response.json())

# --------------------------------------------------
#
#          HELPER METHODS
# --------------------------------------------------


def convert_param_xml_to_ndarray(xml_content: et) -> np.ndarray:
    """
    converts the content of a retrieved XML file into a `numpy.ndarray` type. The output dimension
    is 64 X 64 X 10 which is a data cube with 10 matrix for one image, each matrix for one image
    parameter.

    Note: For the order of image parameters and/or an example of a XML file see:
    http://dmlab.cs.gsu.edu/dmlabapi/ .

    :param xml_content: the content of the retrieved XML file. See the output
                        of `get_aia_imageparam_xml` as an example.
    :return: a `numpy.ndarray` of dimension (x:64) X (y:64) X (image_params:10)
    """
    mat = np.zeros((64, 64, 10))
    x, y, z = 0, 0, 0
    for cell in xml_content:
        x = np.int(cell[0].text)
        y = np.int(cell[1].text)
        z = 0
        for i in np.arange(2, 12):
            mat[x][y][z] = np.float(cell[i].text)
            z = z + 1
    return mat


def convert_param_bulk_xml_to_ndarray(xml_content: et) -> dict:

    d: dict = {}
    for paramframe in xml_content:
        timestamp = paramframe[0].text
        mat = np.zeros((64, 64, 10))
        for cell in paramframe[1:]:  # from 2nd cell onward
            x = np.int(cell[0].text)
            y = np.int(cell[1].text)
            z = 0
            for val in cell[2:]:  # from 3rd cell onward
                mat[x][y][z] = np.float(val.text)
                z = z + 1

        d.update({timestamp: mat})
    return d


def convert_param_json_to_ndarray(json_content: et) -> np.ndarray:
    """
    converts the content of a retrieved JSON file into a `numpy.ndarray` type.
    The output dimension is 64 X 64 X 10 which is a data cube with 10 matrix
    for one image, each matrix for one image parameter.

    Note: For the order of image parameters and/or an example of a JSON file see:
    http://dmlab.cs.gsu.edu/dmlabapi/ .

    :param json_content: the content of the retrieved XML file. See the output
                        of `get_aia_imageparam_json` as an example.
    :return: a `numpy.ndarray` of dimension (x:64) X (y:64) X (image_params:10)
    """
    '''
    TODO Finish this method.
    '''
    pass


def convert_header_xml_to_dict(xml_content: et) -> dict:
    """
    converts the content of a retrieved header in XML format into a `dict' type. The keys are as follows:
    ['X0', 'Y0', 'DSUN', 'R_SUN', 'CDELT', 'QUALITY'].

    :param xml_content: the content of the retrieved XML header file. See the output of `get_aia_imageheader_xml'
                        as an example.
    :return: a `dict' whose values are taken from the XML file.
    """
    keys = ['X0', 'Y0', 'DSUN', 'R_SUN', 'CDELT', 'QUALITY']
    vals = list()
    for cell in xml_content:
        vals.append(cell.text)

    return dict(zip(keys, vals))

# --------------------------------------------------
#
#          QUERY PREPARATION METHODS
# --------------------------------------------------


def prepare_url_get_aia_image_jpeg(starttime: datetime,
                                   aia_wave: (AIA_WAVE, str),
                                   image_size: (IMAGE_SIZE, str)) -> str:
    """
    prepares the query url to communicate with the web api for getting the AIA image corresponding
    to the given start time, wavelength channel, and size.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    :param image_size: size of the image. Either use the class `constants.IMAGE_SIZE` to provide
                       a valid size, or pass in a string from the list: ['2k', '512', '256']
    :return: the prepared query as str.
    """
    time_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')

    prepared_url = URL_STRINGS.aia_image_jpeg_url.format(
        image_size, aia_wave, time_str)
    return prepared_url


def prepare_url_get_aia_imageparam_jpeg(starttime: datetime,
                                        aia_wave: (AIA_WAVE, str),
                                        image_size: (IMAGE_SIZE, str),
                                        param_id: (IMAGE_PARAM, str)) -> str:
    """
    prepares the query url to communicate with the web api for getting the heatmap (JPEG)
    of the given image parameter computed on the 4KX4K AIA image corresponding to the given
    start time, wavelength channel.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
           to provide a valid wavelength, or pass in a string from this list: ['94', '131',
           '171', '193', '211', '304', '335', '1600', '1700']
    :param image_size: size of the output image (heatmap). Either use the class
           `constants.IMAGE_SIZE` to provide a valid size, or pass in a string from the list: [
           '2k', '512', '256']
    :param param_id: id of the image parameters. Either use the class
                     `constants.IMAGE_PARAM` to provide a valid id, or pass in a string from the
                     list: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10]. To know which item
                     corresponds to which parameter, see [dmlab.cs.gsu.aedu/dmlabapi/]
    :return: the prepared query as str.
    """
    time_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')

    prepared_url = URL_STRINGS.aia_imageparam_jpeg_url.format(image_size, aia_wave, time_str,
                                                              param_id)
    return prepared_url


def prepare_url_get_aia_imageparam_xml(starttime: datetime,
                                       aia_wave: (AIA_WAVE, str)) -> str:
    """
    prepares the query url to communicate with the web api for getting the XML of all 10
    image parameters computed on the 4KX4K AIA image corresponding to the given start time,
    wavelength channel.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    :return: the prepared query as str.
    """
    time_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')
    prepared_url = URL_STRINGS.aia_imageparam_xml_url.format(
        aia_wave, time_str)
    return prepared_url


def prepare_url_get_aia_imageparam_json(starttime: datetime,
                                        aia_wave: (AIA_WAVE, str)) -> str:
    time_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')
    prepared_url = URL_STRINGS.aia_imageparam_json_url.format(
        aia_wave, time_str)
    return prepared_url


def prepare_url_get_aia_imageparam_bulk_xml(starttime: datetime,
                                            endtime: datetime,
                                            aia_wave: (AIA_WAVE, str),
                                            limit=100,
                                            offset=0,
                                            step=1) -> str:
    stime_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')
    etime_str = datetime.strftime(endtime, '%Y-%m-%dT%H:%M:%S')
    prepared_url = URL_STRINGS.aia_imageparam_bulk_xml_url.format(
        aia_wave, stime_str, etime_str, limit, offset, step)
    return prepared_url


def prepare_url_get_aia_imageheader_xml(starttime: datetime,
                                        aia_wave: (AIA_WAVE, str)) -> str:
    """
    prepares the query url to communicate with the web api for getting the XML of the header
    information of the AIA image corresponding to the given start time and wavelength channel.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    :return: the prepared query as str.
    """
    time_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')
    prepared_url = URL_STRINGS.aia_image_header_xml_url.format(
        aia_wave, time_str)
    return prepared_url


def prepare_url_get_aia_imageheader_json(starttime: datetime,
                                         aia_wave: (AIA_WAVE, str)) -> str:
    """
    prepares the query url to communicate with the web api for getting the JSON of the header
    information of the AIA image corresponding to the given start time and wavelength channel.

    :param starttime: start time corresponding to the image.
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    :return: the prepared query as str.
    """
    time_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')
    prepared_url = URL_STRINGS.aia_image_header_json_url.format(
        aia_wave, time_str)
    return prepared_url


def prepare_url_get_aia_spatiotemporal_json(starttime: datetime,
                                            endtime: datetime,
                                            table_name: TABLE_NAME,
                                            predicate: PREDICATE,
                                            sort_by,
                                            limit: int,
                                            offset: int) -> str:
    starttime_str = datetime.strftime(starttime, '%Y-%m-%dT%H:%M:%S')
    endtime_str = datetime.strftime(endtime, '%Y-%m-%dT%H:%M:%S')
    prepared_url = URL_STRINGS.aia_spatiotemporal_json_url.format(starttime_str, endtime_str, table_name,
                                                                  predicate, sort_by, limit, offset)
    return prepared_url


# def main():
#     dt = datetime.strptime('2012-02-13T20:10:00', '%Y-%m-%dT%H:%M:%S')
#     aia_wave = AIA_WAVE.AIA_171
#     res = get_aia_spatiotemporal_json(dt, dt, TABLE_NAME.AR, predicate=PREDICATE.INTERSECT, sort_by=dt, limit=1,
#                                                   offset=0)
#     print(res)
#
#
# if __name__ == "__main__":
#     main()
