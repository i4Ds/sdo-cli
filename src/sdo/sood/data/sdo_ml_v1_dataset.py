import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Resize, Normalize, Lambda
import math
import pytorch_lightning as pl


class SDONumpyDataset(Dataset):
    def __init__(
            self,
            base_dir,
            mode="train",
            n_items=None,
            file_pattern="*.npz",
            data_key="x",
            transforms=None):
        """Dataset which loads Numpy npz files
        Args:
            base_dir ([str]): [Directory in which the npz files are.]
            mode (str, optional): [train or val, TODO implement val split]. Defaults to "train".
            n_items ([type], optional): [Number of items in on iteration, by default number of files in the loaded set 
                                        but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
            file_pattern (str, optional): [File pattern of files to load from the base_dir]. Defaults to "*.npz".
            data_key (str, optional): [Data key used to load data from the npz array]. Defaults to 'x'.
            transforms ([type], optional): [Transformations to do after loading the data -> pytorch data transforms]. Defaults to None
        """
        # TODO only load required channels
        self.base_dir = base_dir
        self.file_pattern = file_pattern
        self.files = list(Path(base_dir).glob(f'**/{file_pattern}'))
        self.transforms = transforms
        self.data_key = data_key

        self.data_len = len(self.files)
        print(f"found {len(self.files)} files")
        if n_items is None:
            self.n_items = self.data_len
        else:
            self.n_items = int(n_items)

    def __len__(self):
        return self.n_items

    def __getitem__(self, item):
        if item >= self.n_items:
            raise StopIteration()

        idx = item % self.data_len
        file_path = str(self.files[idx])
        np_arr = np.load(file_path)[self.data_key]  # .astype(np.float64)
        torch_arr = torch.from_numpy(np_arr)
        # convert to 1 x H x W, to be in compatible torchvision format
        torch_arr = torch_arr.unsqueeze(dim=0)
        if self.transforms is not None:
            torch_arr = self.transforms(torch_arr)

        return torch_arr, file_path


# Same preprocess as github.com/i4Ds/SDOBenchmark
CHANNEL_PREPROCESS = {
    "94": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171": {"min": 5, "max": 3500, "scaling": "log10"},
    "193": {"min": 20, "max": 5500, "scaling": "log10"},
    "211": {"min": 7, "max": 3500, "scaling": "log10"},
    "304": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600": {"min": 10, "max": 800, "scaling": "log10"},
    "1700": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "bx": {"min": -250, "max": 250, "scaling": None},
    "by": {"min": -250, "max": 250, "scaling": None},
    "bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=128, channel="171"):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.
    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 128.
        channel (str, optional): [The SDO channel]. Defaults to 171.
    Returns:
        [Transform]
    """

    """
    Apply the normalization necessary for the sdo-dataset. Depending on the channel, it:
      - flips the image vertically
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    :param channel: The kind of data to preprocess
    :param resize: Optional size of image (integer) to resize the image
    :return: a transforms object to preprocess tensors
    """

    #also refer to https://pytorch.org/vision/stable/transforms.html#
    # https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel.lower()]

    if preprocess_config["scaling"] == "log10":
        # TODO why was vflip(x) used here in SolarNet?
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    transforms = Compose(
        [Resize((target_size, target_size)),
         Lambda(lambda_transform),
         Normalize(mean=[mean], std=[std]),
         # required to remove strange distribution of pixels (everything too bright)
         Normalize(mean=(0.5), std=(0.5))
         ]
    )
    return transforms


def get_sdo_ml_v1_dataset(
    base_dir,
    mode="train",
    batch_size=16,
    n_items=None,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    target_size=512,
    file_pattern="*.npz",
    shuffle=False,
):
    """Returns a Pytorch DataLoader which loads an SDONumpyDataset
    Args:
        base_dir ([str]): [Directory in which the npz files are.]
        mode (str, optional): [train or val, TODO implement train val split]. Defaults to "train".
        batch_size (int, optional): [See pytorch DataLoader]. Defaults to 16.
        n_items ([int], optional): [Number of items in the dataset, by default number of files in the loaded set 
                                        but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
        pin_memory (bool, optional): [See pytorch DataLoader]. Defaults to False.
        num_workers (int, optional): [See pytorch DataLoader]. Defaults to 0.
        drop_last (bool, optional): [See pytorch DataLoader]. Defaults to False.
        target_size (int, optional): [New spatial dimension of to which the input data will be transformed]. Defaults to 512.
        file_pattern (str, optional): [File pattern of files to load from the base_dir]. Defaults to "*.npz".
        shuffle (bool, optional): [See pytorch DataLoader]. Defaults to False.
    Returns:
        [DataLoader]: Pytorch data loader which loads an SDONumpyDataset
    """

    transforms = get_default_transforms(
        target_size=target_size, channel=str(171))

    dataset = SDONumpyDataset(
        base_dir=base_dir,
        mode=mode,
        n_items=n_items,
        file_pattern=file_pattern,
        transforms=transforms,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader


class SDOMLv1DataModule(pl.LightningDataModule):
    def __init__(self, base_dir: Path,
                 batch_size: int = 16,
                 n_items: int = None,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 drop_last: bool = False,
                 target_size: int = 512,
                 file_pattern: str = "*.npz",
                 channel: str = "171",
                 shuffle: bool = False):
        """
        Creates a LightningDataModule for the SDO Ml v1 dataset

        Args:
            base_dir ([str]): [Directory in which the npz files are.]
            batch_size (int, optional): [See pytorch DataLoader]. Defaults to 16.
            n_items ([int], optional): [Number of items in the dataset, by default number of files in the loaded set 
                                            but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
            pin_memory (bool, optional): [See pytorch DataLoader]. Defaults to False.
            num_workers (int, optional): [See pytorch DataLoader]. Defaults to 0.
            drop_last (bool, optional): [See pytorch DataLoader]. Defaults to False.
            target_size (int, optional): [New spatial dimension of to which the input data will be transformed]. Defaults to 512.
            file_pattern (str, optional): [File pattern of files to load from the base_dir]. Defaults to "*.npz".
            shuffle (bool, optional): [See pytorch DataLoader]. Defaults to False.
        """
        super().__init__()

        transforms = get_default_transforms(
            target_size=target_size, channel=channel)

        dataset = SDONumpyDataset(
            base_dir=base_dir / Path("train"),
            n_items=n_items,
            file_pattern=file_pattern,
            transforms=transforms,
        )

        self.dataset_test = SDONumpyDataset(
            base_dir=base_dir / Path("test"),
            n_items=n_items,
            file_pattern=file_pattern,
            transforms=transforms,
        )

        # TODO implement temporal split
        num_samples = len(dataset)
        self.dataset_train, self.dataset_val = random_split(
            dataset, [int(num_samples*0.8), int(num_samples * 0.2)])
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)
