import numpy as np
import json
import pandas as pd
from sunpy.net import attrs as a
from sunpy.net import Fido
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, String, MetaData, DateTime, Sequence, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert, JSONB
import logging
import datetime

logging.getLogger('sqlalchemy.engine').setLevel(logging.WARN)
logger = logging.getLogger('HEKEventManager')

# http://solar.stanford.edu/hekwiki/ApplicationProgrammingInterface?action=print
# http://dmlab.cs.gsu.edu/dmlabapi/isd_temporal_queries.html
# https://docs.sunpy.org/en/stable/guide/acquiring_data/hek.html
# https://lmsal.com/hek/api.html
# https://www.lmsal.com/hek/VOEvent_Spec.html
# http://dmlab.cs.gsu.edu/dmlabapi/query/HEK/minevent/spatiotemporal/json/?startTime=2012-01-22T00:00:00&endTime=2012-02-22T00:00:00&xmin=-310&xmax=20&ymin=-200&ymax=20&eventType=ar&sortby=startTime&limit=100&offset=0
# http://dmlab.cs.gsu.edu/dmlabapi/query/HEK/minevent/atemporal/json/?startTime=2012-01-22T00:00:00&endTime=2012-02-22T00:00:00&tableName=ar&predicate=intersects&sortby=startTime&limit=100&offset=0

# https://www.compose.com/articles/using-postgresql-through-sqlalchemy/
# https://github.com/KarthikGP/QueryHEK/blob/master/QueryHek/QueryHek.java
# https://zenodo.org/record/48187
# https://dmlab.cs.gsu.edu/solar/
# https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#insert-on-conflict-upsert


class NpEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super(NpEncoder, self).default(obj)


class HEKEventManager():
    def __init__(self, db_string, *args, **kwargs):
        self.db = create_engine(db_string)
        self.meta = MetaData(self.db)
        self.events_table = Table('hek_events', self.meta,
                                  Column('event_id', Integer, Sequence(
                                      'event_id_seq'), primary_key=True),
                                  Column('event_type', String, nullable=False),
                                  Column('event_starttime',
                                         DateTime, nullable=False),
                                  Column('event_endtime',
                                         DateTime, nullable=False),
                                  Column('obs_observatory', String),
                                  Column('obs_instrument', String),
                                  Column('obs_channelid', String),
                                  Column('kb_archivid', String,
                                         nullable=False),
                                  Column('hpc_bbox', String),
                                  Column('hpc_boundcc', String),
                                  Column('hpc_coord', String),
                                  Column('full_event', JSONB),
                                  UniqueConstraint('kb_archivid', name='uix_kb_archivid'))
        self.events_table.create(checkfirst=True)

    def get_date_ranges(self, start, end, freq="d"):
        # retrieve daily ranges between start and end to not overload the HEK API

        dates = pd.date_range(start=start, end=end,
                              freq=freq).to_pydatetime().tolist()

        dates[0] = start
        if(dates[len(dates) - 1] < end):
            dates.append(end)

        ranges = []
        for i in range(0, len(dates) - 1):
            t_start = dates[i]
            if(i < len(dates) - 1):
                t_end = dates[i + 1]
            else:
                t_end = end

            ranges.append((t_start, t_end))

        return ranges

    def load_and_store_events(self, start: datetime.datetime, end: datetime.datetime, event_type="AR") -> pd.DataFrame:
        logger.info(
            f'loading events between {start} and {end} of type {event_type}')

        event_query = a.hek.EventType(event_type)
        result = Fido.search(a.Time(start, end), event_query)
        col_names = [name for name in result["hek"].colnames if len(
            result["hek"][name].shape) <= 1]
        events_df = result["hek"][col_names].to_pandas()
        logger.info(f"retrieved {len(events_df)} events from HEK")

        # events_df = events_df.astype({"event_peaktime": "datetime64[ns]"})
        # events_df = events_df.set_index("event_peaktime")
        # events_df = events_df.infer_objects()

        with self.db.connect() as conn:
            for idx, event in events_df.iterrows():
                full_event = events_df.iloc[idx].to_dict()
                full_event_json = json.dumps(full_event, cls=NpEncoder)
                insert_statement = insert(self.events_table).values(
                    event_type=event["event_type"],
                    event_starttime=event["event_starttime"],
                    event_endtime=event["event_endtime"],
                    obs_observatory=event["obs_observatory"],
                    obs_instrument=event["obs_instrument"],
                    obs_channelid=event["obs_channelid"],
                    kb_archivid=event["kb_archivid"],
                    hpc_bbox=event["hpc_bbox"],
                    hpc_boundcc=event["hpc_boundcc"],
                    hpc_coord=event["hpc_coord"],
                    # https://www.compose.com/articles/using-json-extensions-in-postgresql-from-python-2/
                    # https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/
                    full_event=json.loads(full_event_json))

                update_dict = {
                    c.name: c for c in insert_statement.excluded if not c.primary_key}
                insert_statement = insert_statement.on_conflict_do_update(
                    index_elements=['kb_archivid'],
                    set_=update_dict)
                conn.execute(insert_statement)

        return events_df

    def load_events_from_hek(self, start: datetime.datetime, end: datetime.datetime, event_type: str):
        """
        Retrieves a set of events from HEK and stores it in the local database
        """
        date_ranges = self.get_date_ranges(start, end)
        total_events = 0
        for t_start, t_end in date_ranges:
            events_df = self.load_and_store_events(t_start, t_end, event_type)
            total_events = total_events + len(events_df)
        logger.info(f"retrieved a total of {total_events} from HEK")

    def read_events(self, start=None, end=None, observatory=None, instrument=None, event_type=None) -> pd.DataFrame:
        with self.db.connect() as conn:
            select_statement = self.events_table.select()

            if start is not None:
                select_statement = select_statement.where(
                    self.events_table.c.event_starttime >= start)
            if end is not None:
                select_statement = select_statement.where(
                    self.events_table.c.event_endtime <= end)

            if instrument is not None:
                select_statement = select_statement.where(
                    self.events_table.c.obs_instrument == instrument)

            if observatory is not None:
                select_statement = select_statement.where(
                    self.events_table.c.obs_observatory == observatory)

            if event_type is not None:
                select_statement = select_statement.where(
                    self.events_table.c.event_type == event_type)

            result_set = conn.execute(select_statement)
            df = pd.DataFrame(result_set)
            if result_set.rowcount > 0:
                df.columns = result_set.keys()

            logger.info(f"retrieved {len(df)} events from local database")
            return df

    def find_events_at(self, timestamp, event_types=None, observatory=None, instrument=None, allowed_time_diff_seconds=30) -> pd.DataFrame:
        with self.db.connect() as conn:
            select_statement = self.events_table.select()
            select_statement = select_statement.where(
                self.events_table.c.event_starttime <= timestamp + datetime.timedelta(seconds=allowed_time_diff_seconds))
            select_statement = select_statement.where(
                self.events_table.c.event_endtime >= timestamp - datetime.timedelta(seconds=allowed_time_diff_seconds))

            if instrument is not None:
                select_statement = select_statement.where(
                    self.events_table.c.obs_instrument == instrument)

            if observatory is not None:
                select_statement = select_statement.where(
                    self.events_table.c.obs_observatory == observatory)

            if event_types is not None:
                select_statement = select_statement.where(
                    self.events_table.c.event_type.in_(event_types))

            result_set = conn.execute(select_statement)
            df = pd.DataFrame(result_set)
            if result_set.rowcount > 0:
                df.columns = result_set.keys()

            logger.info(f"retrieved {len(df)} events from local database")
            return df
