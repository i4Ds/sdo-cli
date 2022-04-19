# adjusted from https://github.com/MIC-DKFZ/mood, Credit: D. Zimmerer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor


def get_transforms(target_size=128):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 128.

    Returns:
        [Transform]
    """

    transforms = Compose([Resize((target_size, target_size)),
                          Grayscale(num_output_channels=1), ToTensor()])

    return transforms


def get_dataset(
    base_dir,
    mode="train",
    batch_size=16,
    n_items=None,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    target_size=128,
    file_pattern="*data.npy",
    do_reshuffle=True,
    slice_offset=0,
    caching=True,
):
    """Returns a Pytorch data loader which loads an ImageFolderDataset

    Args:
        base_dir ([str]): [Directory in which the npy files are.]
        mode (str, optional): [train or val, loads the first 90% for train and 10% for val]. Defaults to "train".
        batch_size (int, optional): [See pytorch DataLoader]. Defaults to 16.
        n_items ([int], optional): [Number of items in on interation, by default number of files in the loaded set 
                                        but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
        pin_memory (bool, optional): [See pytorch DataLoader]. Defaults to False.
        num_workers (int, optional): [See pytorch DataLoader]. Defaults to 0.
        drop_last (bool, optional): [See pytorch DataLoader]. Defaults to False.
        target_size (int, optional): [New spatial dimension of to which the input data will be transformed]. Defaults to 128.
        file_pattern (str, optional): [File pattern of files to load from the base_dir]. Defaults to "*data.npy".
        do_reshuffle (bool, optional): [See pytorch DataLoader]. Defaults to True.


    Returns:
        [DataLoader]: Pytorch data loader which loads an ImageFolderDataset
    """

    transforms = get_transforms(target_size=target_size)

    # https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
    # https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
    data_set = datasets.ImageFolder(base_dir, transforms)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=do_reshuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader
