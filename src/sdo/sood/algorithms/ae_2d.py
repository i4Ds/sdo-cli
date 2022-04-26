# adjusted from https://github.com/MIC-DKFZ/mood, Credit: D. Zimmerer
import os
import time
from math import ceil

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from sdo.sood.data.image_dataset import get_dataset
from sdo.sood.data.path_dataset import ImageFolderWithPaths

from sdo.sood.models.aes import AE
from torchvision.utils import save_image


# TODO remove trixi code
class AE2D:
    def __init__(
        self,
        input_shape,
        lr=1e-4,
        n_epochs=20,
        z_dim=512,
        model_feature_map_sizes=(16, 64, 256, 1024),
        load_path=None,
        log_dir=None,
        logger="visdom",
        print_every_iter=100,
        data_dir=None,
    ):

        self.print_every_iter = print_every_iter
        self.n_epochs = n_epochs
        self.batch_size = input_shape[0]
        self.z_dim = z_dim
        self.input_shape = input_shape
        self.logger = logger
        self.data_dir = data_dir

        log_dict = {}
        if logger is not None:
            log_dict = {
                0: (logger),
            }
        self.tx = PytorchExperimentStub(
            name="ae2d", base_dir=log_dir, config=fn_args_as_config, loggers=log_dict,)

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.model = AE(input_size=input_shape[1:], z_dim=z_dim,
                        fmap_sizes=model_feature_map_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if load_path is not None:
            PytorchExperimentLogger.load_model_static(
                self.model, os.path.join(load_path, "ae_final.pth"))
            time.sleep(5)

    def train(self):

        train_loader = get_dataset(
            base_dir=self.data_dir,
            num_processes=16,
            pin_memory=True,
            batch_size=self.batch_size,
            mode="train",
            target_size=self.input_shape[2],
        )
        val_loader = get_dataset(
            base_dir=self.data_dir,
            num_processes=8,
            pin_memory=True,
            batch_size=self.batch_size,
            mode="val",
            target_size=self.input_shape[2],
        )

        for epoch in range(self.n_epochs):

            # Train
            self.model.train()

            train_loss = 0
            print("\nStart epoch ", epoch)
            data_loader_ = tqdm(enumerate(train_loader))
            for batch_idx, data in data_loader_:
                data = data[0]

                self.optimizer.zero_grad()
                inpt = data.to(self.device)

                inpt_rec = self.model(inpt)

                loss = torch.mean(torch.pow(inpt - inpt_rec, 2))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                if batch_idx % self.print_every_iter == 0:
                    status_str = (
                        f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                        f" ({100.0 * batch_idx / len(train_loader):.0f}%)] Loss: "
                        f"{loss.item() / len(inpt):.6f}"
                    )
                    data_loader_.set_description_str(status_str)

                    cnt = epoch * len(train_loader) + batch_idx
                    self.tx.add_result(
                        loss.item(), name="Train-Loss", tag="Losses", counter=cnt)

                    if self.logger is not None:
                        self.tx.l[0].show_image_grid(
                            inpt, name="Input", image_args={"normalize": True})
                        self.tx.l[0].show_image_grid(
                            inpt_rec, name="Reconstruction", image_args={"normalize": True})

            print(
                f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.4f}")

            # Validate
            self.model.eval()

            val_loss = 0
            with torch.no_grad():
                data_loader_ = tqdm(enumerate(val_loader))
                data_loader_.set_description_str("Validating")
                for i, data in data_loader_:
                    data = data[0]
                    inpt = data.to(self.device)
                    inpt_rec = self.model(inpt)

                    loss = torch.mean(torch.pow(inpt - inpt_rec, 2))
                    val_loss += loss.item()

                self.tx.add_result(
                    val_loss / len(val_loader), name="Val-Loss", tag="Losses", counter=(epoch + 1) * len(train_loader)
                )

            print(
                f"====> Epoch: {epoch} Validation loss: {val_loss / len(val_loader):.4f}")

        self.tx.save_model(self.model, "ae_final")

        time.sleep(10)

    def score_sample(self, data):

        orig_shape = data.shape
        to_transforms = torch.nn.Upsample(
            (self.input_shape[2], self.input_shape[3]), mode="bilinear")

        data_tensor = data
        data_tensor = to_transforms(data_tensor[None])[0]
        slice_scores = []

        for i in range(ceil(orig_shape[0] / self.batch_size)):
            batch = data_tensor[i *
                                self.batch_size: (i + 1) * self.batch_size].unsqueeze(1)
            batch = batch.to(self.device)

            with torch.no_grad():
                batch_rec = self.model(batch)
                loss = torch.mean(
                    torch.pow(batch - batch_rec, 2), dim=(1, 2, 3))

            slice_scores += loss.cpu().tolist()

        return np.max(slice_scores)

    def score_pixels(self, data, index, file_name):

        orig_shape = data.shape
        to_transforms = torch.nn.Upsample(
            (self.input_shape[2], self.input_shape[3]), mode="bilinear")
        from_transforms = torch.nn.Upsample(
            (orig_shape[1], orig_shape[2]), mode="bilinear")

        data_tensor = data
        data_tensor = to_transforms(data_tensor[None])[0]
        target_tensor = torch.zeros_like(data_tensor)

        for i in range(ceil(orig_shape[0] / self.batch_size)):
            batch = data_tensor[i *
                                self.batch_size: (i + 1) * self.batch_size].unsqueeze(1)
            batch = batch.to(self.device)

            batch_rec = self.model(batch)

            loss = torch.pow(batch - batch_rec, 2)[:, 0, :]
            target_tensor[i *
                          self.batch_size: (i + 1) * self.batch_size] = loss.cpu()

        target_tensor = from_transforms(target_tensor[None])[0]
        save_image(target_tensor, file_name, normalize=True)

        return target_tensor.detach().numpy()

    def print(self, *args):
        print(*args)
        self.tx.print(*args)


def main(
    mode="pixel",
    run="train",
    target_size=128,
    batch_size=16,
    n_epochs=20,
    lr=1e-4,
    z_dim=128,
    fmap_sizes=(16, 64, 256, 1024),
    print_every_iter=100,
    load_path=None,
    log_dir=None,
    logger="visdom",
    test_dir=None,
    pred_dir=None,
    data_dir=None,
):

    from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
    from pathlib import Path
    from torch.utils.data import DataLoader

    input_shape = (batch_size, 1, target_size, target_size)

    ae_algo = AE2D(
        input_shape,
        log_dir=log_dir,
        n_epochs=n_epochs,
        lr=lr,
        z_dim=z_dim,
        model_feature_map_sizes=fmap_sizes,
        print_every_iter=print_every_iter,
        load_path=load_path,
        logger=logger,
        data_dir=data_dir,
    )

    if run == "train" or run == "all":
        ae_algo.train()

    if run == "predict" or run == "all":

        if pred_dir is None and log_dir is not None:
            pred_dir = os.path.join(ae_algo.tx.elog.work_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
        elif pred_dir is None and log_dir is None:
            print("Please either give a log/ output dir or a prediction dir")
            exit(0)

        transforms = Compose([Resize((target_size, target_size)),
                              Grayscale(num_output_channels=1), ToTensor()])
        data_set = ImageFolderWithPaths(test_dir, transforms)
        data_loader = DataLoader(data_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)
        for index, (img, label, path) in tqdm(enumerate(data_loader)):
            img = img[0]
            path = Path(path[0])
            if mode == "pixel":
                file_name = os.path.join(pred_dir, path.name)
                pixel_scores = ae_algo.score_pixels(
                    img, index, file_name)

            if mode == "sample":
                sample_score = ae_algo.score_sample(img)
                with open(os.path.join(pred_dir, "predictions.txt"), "a") as target_file:
                    target_file.write(path.name + "," +
                                      str(sample_score) + "\n")


if __name__ == "__main__":

    main()
