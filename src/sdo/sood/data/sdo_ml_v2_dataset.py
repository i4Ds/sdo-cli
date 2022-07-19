from torch import default_generator, randperm, Generator
from typing import (
    Tuple,
    Optional
)
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Resize, Normalize, Lambda
import math
import pytorch_lightning as pl
import zarr
import pandas as pd
import dask.array as da
from typing import Union
from collections import defaultdict
from sdo.sood.data.chunk_sampler import SequenceInChunkSampler
import logging

hmi_date_format = '%Y.%m.%d_%H:%M:%S_TAI'
hmi_channels = ['Bx', 'By', 'Bz']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO manually ignored data points due to invalid samples in SDO ML v2
# remove once https://github.com/SDOML/SDOMLv2/issues/1 is resolved
# channel -> T_OBS
invalid_data = {
    "171A": [
        "2020-02-24T01:48:10.35Z",
        "2020-02-24T01:54:10.35Z",
        "2020-02-24T02:00:10.35Z",
        "2020-02-24T02:06:10.35Z",
        "2020-02-24T02:12:10.35Z",
        "2020-02-24T02:18:10.34Z",
        "2020-02-24T02:24:10.35Z",
        "2020-02-24T02:30:10.35Z",
        "2020-02-24T02:36:10.35Z",
        "2020-02-24T02:42:10.34Z",
        "2020-02-24T02:48:10.35Z",
        "2020-02-24T02:54:10.34Z"
    ]
}


class SDOMLv2NumpyDataset(Dataset):
    def __init__(
            self,
            storage_root: str,
            storage_driver: str,
            year: Union[str, list],
            channel: str,
            start: Union[str, list],
            end:  Union[str, list],
            freq: str,
            irradiance: float,
            irradiance_channel: str,
            goes_cache_dir: str,
            transforms: list,
            cache_max_size: int,
            reduce_memory: bool,
            obs_times: list):
        """Dataset which loads the SDO Ml v2 dataset from a zarr directory.
        """

        if storage_driver == "gcs":
            import gcsfs
            gcs = gcsfs.GCSFileSystem(access="read_only")
            store = gcsfs.GCSMap(storage_root, gcs=gcs, check=False)
        elif storage_driver == "fs":
            store = zarr.DirectoryStore(storage_root)
        else:
            raise f"storage driver {storage_driver} not supported"

        cache = zarr.LRUStoreCache(store, max_size=cache_max_size)
        root = zarr.open(store=cache, mode='r')
        logger.info("discovered the following zarr directory structure")
        logger.info(root.tree())

        if year:
            if type(year) == str:
                by_year = [root[year]]
            elif type(year) == list:
                # TODO this can return None
                by_year = [root.get(y) for y in year]
            else:
                raise ValueError(
                    f"type {type(year)} not supported for argument year, use str or list[str]")
        else:
            by_year = [group for _, group in root.groups()]

        if channel:
            if type(channel) == str:
                data = [group.get(channel) for group in by_year]
            else:
                raise ValueError(
                    f"type {type(channel)} not supported for argument channel, use str or list[str]")
        else:
            data = [g for y in by_year for _, g in y.arrays()]

        all_images = []
        all_attrs = []
        for arr in data:
            images = da.from_array(arr)
            attrs = dict(arr.attrs)

            images, attrs = self.data_quality_check(channel, images, attrs)
            if reduce_memory:
                minimal_keys = ["T_OBS", "WAVELNTH"]
                logger.info(
                    f"reducing memory by retaining only the following attributes {minimal_keys}")
                attrs = {key: attrs[key] for key in minimal_keys}

            if freq:
                images, attrs = self.temporal_downsampling(
                    channel, start, end, freq, images, attrs)
            elif start is not None or end is not None:
                images, attrs = self.temporal_filtering(
                    channel, start, end, images, attrs)

            if irradiance is not None:
                images, attrs = self.irradiance_filtering(
                    channel, irradiance, irradiance_channel, goes_cache_dir, images, attrs)

            if obs_times is not None:
                images, attrs = self.t_obs_filtering(
                    channel, obs_times, images, attrs)
            all_images.append(images)
            all_attrs.append(attrs)

        self.transforms = transforms
        self.all_images = da.concatenate(all_images, axis=0)

        merged_attrs = defaultdict(list)
        for attrs in all_attrs:
            for key, value in attrs.items():
                merged_attrs[key].extend(value)
        self.attrs = merged_attrs
        self.channel = channel

        # call drop invalid attrs again to drop attibutes that are not the same accross years
        self.attrs = self.drop_invalid_attrs(self.all_images, self.attrs)
        self.data_len = len(self.all_images)
        logger.info(
            f"found {len(self.all_images)} images")

    def drop_invalid_attrs(self, images, attrs):
        keys_to_remove = set()
        for key, values in attrs.items():
            if len(values) != len(images):
                logger.warn(
                    f"attr {key} does not have the same length ({len(values)}) as the data ({len(images)}), dropping it...")
                keys_to_remove.add(key)
                continue

            for value in values:
                if value is None:
                    logger.warn(
                        f"attr {key} has None values, dropping it...")
                    keys_to_remove.add(key)
                    break

        for key in keys_to_remove:
            del attrs[key]

        return attrs

    def irradiance_filtering(self, channel, irradiance, irradiance_channel, goes_cache_dir, images, attrs):
        from sdo.data_loader.goes.cache import GOESCache
        goes_cache = GOESCache(goes_cache_dir)
        df_time = self.obs_time_as_df(channel, attrs)
        for i in range(len(df_time.index)):
            ts = df_time.loc[i]["Time"]
            try:
                goes_at = goes_cache.get_goes_at(ts.replace(tzinfo=None))
            except ValueError:
                logger.warn(f"no GOES value found at {ts}, setting to np.nan")
                goes_at = {"xrsa": np.nan, "xrsb": np.nan}

            df_time.loc[i, 'xrsa'] = goes_at["xrsa"]
            df_time.loc[i, 'xrsb'] = goes_at["xrsb"]
        irr_filter = (df_time[irradiance_channel] <= irradiance)
        irr_index = df_time['Time'].index[irr_filter].tolist()
        images = images[irr_index, :, :]
        attrs = {keys: [values[idx] for idx in irr_index]
                 for keys, values in attrs.items()}

        return images, attrs

    def t_obs_filtering(self, channel, obs_times, images, attrs):
        logger.info(
            f"filtering for {len(obs_times)} obs times")

        obs_times = pd.to_datetime(obs_times, format=None, utc=True)
        df_time = self.obs_time_as_df(channel, attrs)
        filter = np.isin(df_time['Time'], obs_times)

        time_index = df_time['Time'].index[filter].tolist()
        images = images[time_index, :, :]
        attrs = {keys: [values[idx] for idx in time_index]
                 for keys, values in attrs.items()}
        return images, attrs

    def temporal_filtering(self, channel, start, end, images, attrs):
        logger.info(
            f"filtering data between {start} and {end}")

        df_time = self.obs_time_as_df(channel, attrs)

        if start is not None and end is not None:
            filter = (df_time['Time'] >= start) & (
                df_time['Time'] <= end)
        elif start is not None:
            filter = (df_time['Time'] >= start)
        else:
            filter = (df_time['Time'] <= end)

        time_index = df_time['Time'].index[filter].tolist()
        images = images[time_index, :, :]
        attrs = {keys: [values[idx] for idx in time_index]
                 for keys, values in attrs.items()}
        return images, attrs

    def data_quality_check(self, channel, images, attrs):
        logger.info(
            f"checking data quality and removing invalid samples for {len(images)} images")

        attrs = self.drop_invalid_attrs(images, attrs)

        invalid_times = invalid_data.get(channel)
        if invalid_times:
            t_obs = np.array(attrs["T_OBS"])
            df_time = pd.DataFrame(t_obs, index=np.arange(
                np.shape(t_obs)[0]), columns=["Time"])
            # equivalent to (but faster than) np.invert(np.isin(a, b))
            filter = np.isin(df_time['Time'], invalid_times, invert=True)

            clean_index = df_time['Time'].index[filter].tolist()
            images = images[clean_index, :, :]
            try:
                attrs = {keys: [values[idx] for idx in clean_index]
                         for keys, values in attrs.items()}
            except Exception as e:
                attr_length = len(attrs["T_OBS"])
                logger.error(
                    f"could not apply data cleaning with {clean_index} for {attr_length}", e)
                raise e
        return images, attrs

    def temporal_downsampling(self, channel, start, end, freq, images, attrs):
        logger.info(
            f"applying temporal downsampling to freq {freq} between {start} and {end}")

        df_time = self.obs_time_as_df(channel, attrs)

        # select times at a frequency of freq (e.g. 12T)
        selected_times = pd.date_range(
            start=start, end=end, freq=freq, tz="UTC"
        )
        selected_index = []
        for i in selected_times:
            selected_index.append(np.argmin(abs(df_time["Time"] - i)))
        time_index = [x for x in selected_index if x > 0]
        images = images[time_index, :, :]
        attrs = {keys: [values[idx] for idx in time_index]
                 for keys, values in attrs.items()}
        return images, attrs

    def obs_time_as_df(self, channel, attrs):
        t_obs = np.array(attrs["T_OBS"])
        df_time = pd.DataFrame(t_obs, index=np.arange(
            np.shape(t_obs)[0]), columns=["Time"])
        # NOTE for HMI the date format is different 2010.05.01_00:12:04_TAI
        format = None
        if channel in hmi_channels:
            format = hmi_date_format
        df_time["Time"] = pd.to_datetime(
            df_time["Time"], format=format, utc=True)

        return df_time

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        try:
            image = self.all_images[idx, :, :]

            attrs = dict([(key, self.attrs[key][idx])
                          for key in self.attrs.keys()])

            # Pytorch does not support NoneType items
            attrs = {k: v for k, v in attrs.items() if v is not None}

            torch_arr = torch.from_numpy(np.array(image))
            # convert to 1 x H x W, to be in compatible torchvision format
            torch_arr = torch_arr.unsqueeze(dim=0)
            if self.transforms is not None:
                torch_arr = self.transforms(torch_arr)

            return torch_arr, attrs
        except Exception as e:
            logger.error(f"could not get image for index {idx}", e)
            raise e


# TODO are these values still applicable for the new correction factors?
# Same preprocess as github.com/i4Ds/SDOBenchmark
# Notably in SDOBenchmark some of the preprocessing seems to be different from the preprocessing in Helioviewer
# https://github.com/Helioviewer-Project/jp2gen/blob/master/idl/sdo/aia/hvs_default_aia.pro
# e.g. AIA 171 has log10 scaling in SDO benchmark and sqrt scaling in Helioviewer
CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "Bx": {"min": -250, "max": 250, "scaling": None},
    "By": {"min": -250, "max": 250, "scaling": None},
    "Bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=256, channel="171", mask_limb=False, radius_scale_factor=1.0):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - masks the limb with 0s
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 256.
        channel (str, optional): [The SDO channel]. Defaults to 171.
        mask_limb (bool, optional): [Whether to mask the limb]. Defaults to False.
        radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the limb]. Defaults to 1.0.
    Returns:
        [Transform]
    """

    transforms = []

    # also refer to
    # https://pytorch.org/vision/stable/transforms.html
    # https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO does it make sense to use vflip(x) in order to align the solar North as in JHelioviewer?
        # otherwise this has to be done during inference
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - \
            math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    def limb_mask_transform(x):
        h, w = x.shape[1], x.shape[2]  # C x H x W

        # fixed disk size of Rs of 976 arcsec, pixel size in the scaled image (512x512) is ~4.8 arcsec
        original_resolution = 4096
        scaled_resolution = h
        pixel_size_original = 0.6
        radius_arcsec = 976.0
        radius = (radius_arcsec / pixel_size_original) / \
            original_resolution * scaled_resolution

        mask = create_circular_mask(
            h, w, radius=radius*radius_scale_factor)
        mask = torch.as_tensor(mask, device=x.device)
        return torch.where(mask, x, torch.tensor(0.0))

    if mask_limb:
        transforms.append(Lambda(lambda x: limb_mask_transform(x)))

    transforms.append(Resize((target_size, target_size)))
    # TODO find out if these transforms make sense
    transforms.append(Lambda(lambda x: lambda_transform(x)))
    transforms.append(Normalize(mean=[mean], std=[std]))
    # required to remove strange distribution of pixels (everything too bright)
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)


def create_circular_mask(h, w, center=None, radius=None):
    # TODO investigate the use of a circular mask to prevent focussing to much on the limb
    # https://gitlab.com/jdonzallaz/solarnet-app/-/blob/master/src/prediction.py#L9

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


class SDOMLv2DataModule(pl.LightningDataModule):
    def __init__(self,
                 storage_root: str = "fdl-sdoml-v2/sdomlv2_small.zarr/",
                 storage_driver: str = "gcs",
                 batch_size: int = 16,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 drop_last: bool = False,
                 prefetch_factor: int = 8,
                 cache_max_size: int = 1*1024*1024*2014,
                 target_size: int = 256,
                 channel: str = "171A",
                 train_year: Union[str, list] = "2010",
                 train_start=None,
                 train_end=None,
                 test_year: Union[str, list] = "2010",
                 test_start=None,
                 test_end=None,
                 freq=None,
                 obs_times: list = None,
                 irradiance: float = None,
                 irradiance_channel: str = "xrsb",
                 goes_cache_dir: str = None,
                 shuffle: bool = False,
                 train_val_split_strategy: str = "temporal",
                 train_val_split_ratio: float = 0.7,
                 train_val_split_temporal_chunk_size: str = "14d",
                 skip_train_val: bool = False,
                 skip_test: bool = False,
                 sampling_strategy: str = "chunk",
                 mask_limb: bool = False,
                 mask_limb_radius_scale_factor: float = 1.0,
                 reduce_memory: bool = False
                 ):
        """
        Creates a LightningDataModule for the SDO Ml v2 dataset

        The header information can be retrieved as the second argument when enumerating the loader

        >>> loader = SDOMLv2DataModule(storage_root="fdl-sdoml-v2/sdomlv2_small.zarr/").train_dataloader()
        >>> for batch_idx, batch in enumerate(loader):
        >>>     X, headers  = batch

        Args:
            storage_root (str): [Root path containing the zarr archives]
            storage_driver(str, optional): [Storage driver used to load the data. Either 'gcs' (Google Storage Bucket) or 'fs' (local file system)]. Defaults to gcs.
            batch_size (int, optional): [See pytorch DataLoader]. Defaults to 16.
            pin_memory (bool, optional): [See pytorch DataLoader]. Defaults to False.
            num_workers (int, optional): [See pytorch DataLoader]. Defaults to 0.
            drop_last (bool, optional): [See pytorch DataLoader]. Defaults to False.
            cache_max_size (int, optional): The maximum size that the cache may grow to, in number of bytes. Defaults to 1GB.
            target_size (int, optional): [New spatial dimension of to which the input data will be transformed]. Defaults to 256.
            channel (str, optional): [Channel name that should be used. If None all available channels will be used.]. Defaults to "171A".
            train_year (str|list[str], optional): [Allows to prefilter the dataset by year. Can be a string or list of strings. If None all available years will be used.]. Defaults to "2010".
            train_start (str, optional): [Allows to restrict the dataset temporally]. Defaults to None.
            train_end (str, optional): [Allows to restrict the dataset temporally]. Defaults to None.
            test_year (str|list[str], optional): [Allows to prefilter the dataset by year. Can be a string or list of strings. If None all available years will be used.]. Defaults to "2010".
            test_start (str, optional): [Allows to restrict the dataset temporally]. Defaults to None.
            test_end (str, optional): [Allows to restrict the dataset temporally]. Defaults to None.
            freq (str, optional): [Allows to downsample the dataset temporally, should be bigger than the min interval for the observed channel. When using freq, start and end should also be specified for train and test]. Defaults to None.
            obs_times (list, optional): [Allows to filter for specific observation times]. Defaults to None.
            irradiance (float, optional): Allows to filter by irradiance (all images with smaller or equal values will be in the resulting dataset). Defaults to None.
            irradiance_channel (str, optional): Either xrsa or xrsb. Refer to the GOES documentation. Defaults to "xrsb".
            goes_cache_dir (str, optional): Path to the cached GOES values (downloaded with sdo-cli goes download --help). Defaults to None.
            shuffle (bool, optional): [See pytorch DataLoader]. Defaults to False.
            train_val_split_strategy (str, optional): [Strategy for the train-validation split. Either 'temporal' or 'random']. Defaults to "temporal".
            train_val_split_ratio (float, optional): [Split-ratio for the train-validation split]. Defaults to 0.7.
            train_val_split_temporal_chunk_size (str, optional): [Temporal chunks for the train-validation splits]. Defaults to "14d".
            skip_train_val (bool, optional): [Whether to skip loading the train and validation sets]: Defaults to False.
            skip_test (bool, optional): [Whether to skip loading the test set]: Defaults to False.
            sampling_strategy (str, optional): [Which sampling strategy to use (chunk or default)]: Defaults to "chunk".
            mask_limb (bool, optional): [Whether to mask the solar limb]. Defaults to False.
            mask_limb_radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the solar limb]. Defaults to 1.0.
            reduce_memory (bool, optional): [Reduces memory by only retaining required attributes]. Defaults to False.
        """

        super().__init__()

        transforms = get_default_transforms(
            target_size=target_size, channel=channel, mask_limb=mask_limb, radius_scale_factor=mask_limb_radius_scale_factor)

        # TODO this can be achieved with setup()
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.hooks.DataHooks.html#pytorch_lightning.core.hooks.DataHooks.setup
        if not skip_test:
            self.dataset_test = SDOMLv2NumpyDataset(
                storage_root=storage_root,
                storage_driver=storage_driver,
                cache_max_size=cache_max_size,
                year=test_year,
                start=test_start,
                end=test_end,
                freq=freq,
                obs_times=obs_times,
                irradiance=irradiance,
                irradiance_channel=irradiance_channel,
                goes_cache_dir=goes_cache_dir,
                channel=channel,
                transforms=transforms,
                reduce_memory=reduce_memory
            )

        if not skip_train_val:
            dataset = SDOMLv2NumpyDataset(
                storage_root=storage_root,
                storage_driver=storage_driver,
                cache_max_size=cache_max_size,
                year=train_year,
                start=train_start,
                end=train_end,
                freq=freq,
                obs_times=obs_times,
                irradiance=irradiance,
                irradiance_channel=irradiance_channel,
                goes_cache_dir=goes_cache_dir,
                channel=channel,
                transforms=transforms,
                reduce_memory=reduce_memory
            )
            if train_val_split_strategy == "random":
                num_samples = len(dataset)
                splits = [int(math.floor(num_samples*train_val_split_ratio)),
                          int(math.ceil(num_samples * (1 - train_val_split_ratio)))]
                logger.info(
                    f"splitting datatset with {num_samples} into {splits}")
                self.dataset_train, self.dataset_val = random_split(
                    dataset, splits)
            elif train_val_split_strategy == "temporal":
                self.dataset_train, self.dataset_val = temporal_train_val_split(
                    dataset, split_ratio=train_val_split_ratio, temporal_chunk_size=train_val_split_temporal_chunk_size)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.prefetch_factor = prefetch_factor
        self.sampling_strategy = sampling_strategy

    def train_dataloader(self):
        sampler = None
        if self.sampling_strategy == "chunk":
            sampler = SequenceInChunkSampler(self.dataset_train)
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          prefetch_factor=self.prefetch_factor,
                          sampler=sampler)

    def val_dataloader(self):
        sampler = None
        if self.sampling_strategy == "chunk":
            sampler = SequenceInChunkSampler(self.dataset_val)
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=math.ceil(self.num_workers/2),
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          prefetch_factor=self.prefetch_factor,
                          sampler=sampler)

    def test_dataloader(self):
        sampler = None
        if self.sampling_strategy == "chunk":
            sampler = SequenceInChunkSampler(self.dataset_test)
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          prefetch_factor=self.prefetch_factor,
                          sampler=sampler)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          prefetch_factor=self.prefetch_factor)


def temporal_train_val_split(dataset: SDOMLv2NumpyDataset, split_ratio=0.7, temporal_chunk_size="14d",
                             generator: Optional[Generator] = default_generator) -> Tuple[Subset, Subset]:
    r"""
    Temporally split a dataset into non-overlapping temporal chunks and compose the chunks to new datasets of the given split ratio.
    The split ratio is an approximation as all data in a temporal chunk will end up in either the train or validation dataset.

    Optionally fix the generator for reproducible results, e.g.:

    >>> temporal_split(sdo_ml_dataset, split_ratio=0.8, temporal_chunk_size="6h", generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): SDOML v2 Dataset to be split
        split_ratio (float): train-validation split ratio
        temporal_chunk_size (freq): size of the temporal chunks as a frequency
        generator (Generator): Generator used for the random permutation.
    """

    assert hasattr(
        dataset, 'attrs'), "dataset should have an attrs property"

    format = None
    if dataset.channel in hmi_channels:
        format = hmi_date_format

    t_obs = np.array(dataset.attrs["T_OBS"])
    data = {"Time": t_obs, "Index": np.arange(np.shape(t_obs)[0])}
    df_time = pd.DataFrame(data, index=np.arange(
        np.shape(t_obs)[0]), columns=["Time", "Index"])
    df_time["Time"] = pd.to_datetime(
        df_time["Time"], format=format, utc=True)
    df_time = df_time.set_index('Time')
    grouped = df_time.groupby(pd.Grouper(freq=temporal_chunk_size))

    group_indices = randperm(len(grouped), generator=generator).tolist()
    num_chunks = len(group_indices)
    train_size = int(math.ceil(num_chunks*split_ratio))
    logger.info(
        f"Selecting groups for train-validation split. Number of groups {num_chunks}, number of groups for training {train_size}, number of groups for validation {num_chunks - train_size}")
    train_indices = []
    val_indices = []
    groups = list(grouped)
    for i in group_indices[0:train_size]:
        _, group = groups[i]
        for _, row in group.iterrows():
            train_indices.append(row["Index"])

    for i in group_indices[train_size:]:
        _, group = groups[i]
        for _, row in group.iterrows():
            val_indices.append(row["Index"])

    logger.info(
        f"splitting Dataset into two subsets. Train size {len(train_indices)}, validation size {len(val_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
