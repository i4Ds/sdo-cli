# Credit A.Ahmadzadeh, https://bitbucket.org/gsudmlab/imageparams_api/src/master/
from sdo.data_loader.image_param.IP_CONSTANTS.CONSTANTS import *
from datetime import datetime
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from shapely import wkt


class SpatioTemporalEvent:
    """
    This class encapsulates the retrieved data using the following method:

        `wrappers.get_aia_spatiotemporal_json'

    For more info, visit: 'http://dmlab.cs.gsu.edu/dmlabapi/isd_temporal_queries.html'
    """

    def __init__(self,
                 event_type: EVENT_TYPE,
                 start_time: datetime,
                 end_time: datetime,
                 hpc_coord: (Point, str),
                 hpc_bbox: (Polygon, str),
                 hpc_boundcc: (Polygon, str),
                 kb_archivid: str):
        """
        :param event_type: The event for which the results are returned.
        :param start_time: Start time of the event type.
        :param end_time: End Time of the event Type.
        :param hpc_coord: coordinates of the center of the bounding box.
        :param hpc_bbox: bounding box of the polygon.
        :param hpc_boundcc: polygon of the detected event (if present).
        :param kb_archivid: Unique id for each event type.
        """
        self.event_type: EVENT_TYPE = event_type
        self.start_time: datetime = start_time
        self.end_time: datetime = end_time
        self.hpc_coord: (Point, str) = hpc_coord
        self.hpc_bbox: (Polygon, str) = hpc_bbox
        self.hpc_boundcc: (Polygon, str) = hpc_boundcc
        self.kb_archivid: str = kb_archivid

    @classmethod
    def from_dict(cls, instance: dict):
        obj = cls.__new__(cls)
        super(SpatioTemporalEvent, obj).__init__()
        obj.event_type = instance['eventtype']
        obj.start_time = instance['starttime']
        obj.end_time = instance['endtime']
        obj.hpc_coord = instance['hpc_coord']
        obj.hpc_bbox = instance['hpc_bbox']
        obj.hpc_boundcc = instance['hpc_boundcc']
        obj.kb_archivid = instance['kb_archivid']
        return obj

    @property
    def event_type(self):
        return self.__event_type

    @event_type.setter
    def event_type(self, event_type):
        if isinstance(event_type, EVENT_TYPE):
            self.__event_type = event_type
        elif isinstance(event_type, str):
            self.__event_type = EVENT_TYPE.convert(event_type)

    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        if isinstance(start_time, datetime):
            self.__start_time = start_time
        elif isinstance(start_time, str):
            self.__start_time = datetime.strptime(
                start_time, '%Y-%m-%d %H:%M:%S')
        else:
            raise AttributeError

    @property
    def end_time(self):
        return self.__end_time

    @end_time.setter
    def end_time(self, end_time):
        if isinstance(end_time, datetime):
            self.__end_time = end_time
        elif isinstance(end_time, str):
            self.__end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        else:
            raise AttributeError

    @property
    def hpc_coord(self):
        return self.__hpc_coord

    @hpc_coord.setter
    def hpc_coord(self, hpc_coord):
        if isinstance(hpc_coord, Point):
            self.__hpc_coord = hpc_coord
        elif isinstance(hpc_coord, str):
            self.__hpc_coord = wkt.loads(hpc_coord)
        else:
            raise AttributeError

    @property
    def hpc_bbox(self):
        return self.__hpc_bbox

    @hpc_bbox.setter
    def hpc_bbox(self, hpc_bbox):
        if isinstance(hpc_bbox, Polygon):
            self.__hpc_bbox = hpc_bbox
        elif isinstance(hpc_bbox, str):
            self.__hpc_bbox = wkt.loads(hpc_bbox)
        else:
            raise AttributeError

    @property
    def hpc_boundcc(self):
        return self.__hpc_boundcc

    @hpc_boundcc.setter
    def hpc_boundcc(self, hpc_boundcc):
        if isinstance(hpc_boundcc, Polygon):
            self.__hpc_boundcc = hpc_boundcc
        elif isinstance(hpc_boundcc, str):
            self.__hpc_boundcc = wkt.loads(hpc_boundcc)
        else:
            raise AttributeError

    @property
    def kb_archivid(self):
        return self.__kb_archivid

    @kb_archivid.setter
    def kb_archivid(self, kb_archivid):
        self.__kb_archivid = kb_archivid

    def to_dict(self):
        dict = {'event_type': self.event_type,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'hpc_coord': self.hpc_coord,
                'hpc_bbox': self.hpc_bbox,
                'hpc_boundcc': self.hpc_boundcc,
                'kb_archivid': self.kb_archivid
                }
        return dict
