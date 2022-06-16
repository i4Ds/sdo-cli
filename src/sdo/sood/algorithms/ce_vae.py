# adjusted from https://github.com/MIC-DKFZ/mood, Credit: D. Zimmerer
# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
import json
from dateutil.parser import parse
import datetime
import os
import sys
from math import ceil
from pathlib import Path
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as dist
import wandb
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary)
from pytorch_lightning.loggers import WandbLogger
from sdo.sood.data.image_dataset import get_dataset
from sdo.sood.data.path_dataset import ImageFolderWithPaths
from sdo.sood.data.sdo_ml_v1_dataset import SDOMLv1DataModule
from sdo.sood.data.sdo_ml_v2_dataset import SDOMLv2DataModule
from sdo.sood.models.aes import VAE
from sdo.sood.util.ce_noise import get_square_mask, normalize, smooth_tensor
from sdo.sood.util.utils import (get_smooth_image_gradient, read_config,
                                 tensor_to_image)
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ceVAE(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        lr,
        z_dim: int,
        model_feature_map_sizes,
        use_geco: bool,
        beta,
        ce_factor,
        score_mode,
        mode,
        print_every_iter
    ):
        super().__init__()

        self.score_mode = score_mode
        self.mode = mode
        self.ce_factor = ce_factor
        self.beta = beta
        self.print_every_iter = print_every_iter
        self.batch_size = input_shape[0]
        self.z_dim = z_dim
        self.use_geco = use_geco
        self.input_shape = input_shape
        self.lr = lr
        self.vae_loss_ema = 1
        self.theta = 1

        self.model = VAE(
            input_size=input_shape[1:], z_dim=z_dim, fmap_sizes=model_feature_map_sizes)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def pixel_mode(self):
        self.mode = "pixel"

    def sample_mode(self):
        self.mode = "sample"

    def training_step(self, batch, batch_idx):
        x, attrs = batch  # only inputs no labels

        # VAE Part
        loss_vae = 0
        loss_vae_kl = 0
        loss_vae_rec = 0
        try:
            if self.ce_factor < 1:
                x_rec_vae, z_dist, = self.model(x)

                if self.beta > 0:
                    loss_vae_kl = self.kl_loss_fn(z_dist) * self.beta

                loss_vae_rec = self.rec_loss_fn(x_rec_vae, x)
                loss_vae = loss_vae_kl + loss_vae_rec * self.theta

            # CE Part
            loss_ce = 0
            if self.ce_factor > 0:
                ce_tensor = get_square_mask(
                    x.shape,
                    square_size=(0, np.max(self.input_shape[2:]) // 2),
                    noise_val=(torch.min(x).item(),
                               torch.max(x).item()),
                    n_squares=(0, 3),
                )
                ce_tensor = torch.from_numpy(ce_tensor).float()
                ce_tensor = ce_tensor.type_as(x)
                inpt_noisy = torch.where(ce_tensor != 0, ce_tensor, x)
                x_rec_ce, _ = self.model(inpt_noisy)
                rec_loss_ce = self.rec_loss_fn(x_rec_ce, x)
                loss_ce = rec_loss_ce

            loss = (1.0 - self.ce_factor) * \
                loss_vae + self.ce_factor * loss_ce

            # Generalized ELBO with Constrained Optimization
            if self.use_geco and self.ce_factor < 1:
                g_goal = 0.1
                g_lr = 1e-4
                self.vae_loss_ema = (1.0 - 0.9) * \
                    loss_vae_rec + 0.9 * self.vae_loss_ema
                self.theta = self.geco_beta_update(
                    self.theta, self.vae_loss_ema, g_goal, g_lr, speedup=2)

                self.log('geco_theta', self.theta)
                self.log('geco_vae_loss_ema', self.vae_loss_ema)
        except Exception as e:
            logger.error(
                f"exception during training step for data {x} with attrs {attrs}", e)
            raise e

        if batch_idx % self.print_every_iter == 0:
            sample_images = []
            if self.ce_factor < 1:
                sample_images.append(wandb.Image(tensor_to_image(
                    x, normalize=False), caption="Input-VAE"))
                sample_images.append(wandb.Image(tensor_to_image(
                    x_rec_vae, normalize=False), caption="Output-VAE"))
            if self.ce_factor > 0:
                sample_images.append(wandb.Image(tensor_to_image(
                    inpt_noisy, normalize=False), caption="Input-CE"))
                sample_images.append(wandb.Image(tensor_to_image(
                    x_rec_ce, normalize=False), caption="Output-CE"))

            self.logger.experiment.log(
                {"train_images": sample_images})

        self.log('train_loss', loss)
        self.log('train_loss_vae', loss_vae)
        self.log('train_loss_ce', loss_ce)

        return {"loss": loss, "loss_vae": loss_vae, "loss_ce": loss_ce, "loss_vae_kl": loss_vae_kl, "loss_vae_rec": loss_vae_rec}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_rec, z_dist = self.model(x, sample=False)
        kl_loss = 0
        if self.beta > 0:
            kl_loss = self.kl_loss_fn(z_dist) * self.beta
        rec_loss = self.rec_loss_fn(x_rec, x)
        loss = kl_loss + rec_loss * self.theta

        self.log('val_loss', loss)
        self.log('val_loss_kl', kl_loss)
        self.log('val_loss_rec', rec_loss)

        # log image for every 10th validation batch
        if batch_idx % 10 == 0:
            self.logger.experiment.log(
                {"val_images": [wandb.Image(tensor_to_image(
                    x, normalize=False), caption="Input"), wandb.Image(tensor_to_image(
                        x_rec, normalize=False), caption="Output")]})
        return loss

    def test_step(self, batch, batch_idx):
        logger.warn("test step not defined")
        pass

    def forward(self, x):
        if self.mode == "sample":
            return self.score_sample(x)
        elif self.mode == "pixel":
            return self.score_pixels(x)
        else:
            raise ValueError(f"invalid mode {self.mode}")

    def generate(self, n_samples=16, mu=None, std=None):
        if mu is None:
            mu = torch.zeros_like(torch.empty(self.z_dim, 1, 1))
        if std is None:
            std = torch.ones_like(torch.empty(self.z_dim, 1, 1))

        p = torch.distributions.Normal(mu, std)
        z = p.rsample((n_samples,))

        with torch.no_grad():
            pred = self.model.decode(z.to(self.device)).cpu()

        file_name = Path(self.work_dir) / Path("generated") / \
            (datetime.datetime.now().isoformat() + "_generated.jpeg")
        save_image(pred, file_name, normalize=True)

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
            # batch = batch * 2 - 1

            with torch.no_grad():
                inpt = batch.to(self.device).float()
                x_rec, z_dist = self.model(inpt, sample=False)
                kl_loss = self.kl_loss_fn(z_dist, sum_samples=False)
                rec_loss = self.rec_loss_fn(x_rec, inpt, sum_samples=False)
                img_scores = kl_loss * self.beta + rec_loss * self.theta

            slice_scores += img_scores.cpu().tolist()

        return np.max(slice_scores)

    def score_pixels(self, data):
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

            inpt = batch.to(self.device).float()
            x_rec, z_dist = self.model(inpt, sample=False)

            if self.score_mode == "combi":
                rec = torch.pow((x_rec - inpt), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)

                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    loss = self.kl_loss_fn(z_d)
                    return loss

                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=inpt, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(loss_grad_kl, dim=1, keepdim=True)

                pixel_scores = smooth_tensor(
                    normalize(loss_grad_kl), kernel_size=8) * rec

            elif self.score_mode == "rec":
                rec = torch.pow((x_rec - inpt), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)
                pixel_scores = rec

            elif self.score_mode == "grad":
                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    kl_loss_ = self.kl_loss_fn(z_d)
                    rec_loss_ = self.rec_loss_fn(x_r, x)
                    loss_ = kl_loss_ * self.beta + rec_loss_ * self.theta
                    return torch.mean(loss_)

                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=inpt, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(loss_grad_kl, dim=1, keepdim=True)

                pixel_scores = smooth_tensor(
                    normalize(loss_grad_kl), kernel_size=8)

            # save_image_grid(inpt, name="Input", save_dir=self.work_dir, image_args={
            #     "normalize": True}, n_iter=index)
            # save_image_grid(x_rec, name="Output", save_dir=self.work_dir, image_args={
            #     "normalize": True}, n_iter=index)
            # save_image_grid(pixel_scores, name="Scores", save_dir=self.work_dir, image_args={
            #     "normalize": True}, n_iter=index)

            target_tensor[i * self.batch_size: (
                i + 1) * self.batch_size] = pixel_scores.detach().cpu()[:, 0, :]

        target_tensor = from_transforms(target_tensor[None])[0]
        # TODO rather normalize over the whole dataset rather than a single image
        # save_image(target_tensor, file_name, normalize=True)

        return target_tensor

    @staticmethod
    def kl_loss_fn(z_post, sum_samples=True, correct=False):
        z_prior = dist.Normal(0, 1.0)
        kl_div = dist.kl_divergence(z_post, z_prior)
        if correct:
            kl_div = torch.sum(kl_div, dim=(1, 2, 3))
        else:
            kl_div = torch.mean(kl_div, dim=(1, 2, 3))
        if sum_samples:
            return torch.mean(kl_div)
        else:
            return kl_div

    @staticmethod
    def rec_loss_fn(recon_x, x, sum_samples=True, correct=False):
        if correct:
            x_dist = dist.Laplace(recon_x, 1.0)
            log_p_x_z = x_dist.log_prob(x)
            log_p_x_z = torch.sum(log_p_x_z, dim=(1, 2, 3))
        else:
            log_p_x_z = -torch.abs(recon_x - x)
            log_p_x_z = torch.mean(log_p_x_z, dim=(1, 2, 3))
        if sum_samples:
            return -torch.mean(log_p_x_z)
        else:
            return -log_p_x_z

    @staticmethod
    def get_inpt_grad(model, inpt, err_fn):
        model.zero_grad()
        inpt = inpt.detach()
        inpt.requires_grad = True

        err = err_fn(inpt)
        err.backward()

        grad = inpt.grad.detach()

        model.zero_grad()

        return torch.abs(grad.detach())

    @staticmethod
    def geco_beta_update(beta, error_ema, goal, step_size, min_clamp=1e-10, max_clamp=1e4, speedup=None):
        constraint = (error_ema - goal).detach()
        if speedup is not None and constraint > 0.0:
            beta = beta * torch.exp(speedup * step_size * constraint)
        else:
            beta = beta * torch.exp(step_size * constraint)
        if min_clamp is not None:
            beta = np.max((beta.item(), min_clamp))
        if max_clamp is not None:
            beta = np.min((beta.item(), max_clamp))
        return beta

    @staticmethod
    def get_ema(new, old, alpha):
        if old is None:
            return new
        return (1.0 - alpha) * new + alpha * old


folder_time_format = "%Y%m%d-%H%M%S"


def main(
    run: str = "train",
    config_file: Path = Path("./config/defaults.yaml")
):
    config = read_config(config_file)
    logger.info("found config")
    logger.info(json.dumps(config, indent=2))

    current_run_name = f"{datetime.datetime.now().strftime(folder_time_format)}_cevae"
    work_dir = Path(config.log_dir.value) / Path(current_run_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    input_shape = (config.data.batch_size.value, 1,
                   config.model.target_size.value, config.model.target_size.value)

    if config.model.load_path.value is not None:
        cevae_algo = ceVAE.load_from_checkpoint(
            config.model.load_path.value, mode=config.predict.mode.value)
    else:
        cevae_algo = ceVAE(
            input_shape,
            lr=config.train.lr.value,
            z_dim=config.model.z_dim.value,
            model_feature_map_sizes=config.model.fmap_sizes.value,
            use_geco=config.train.use_geco.value,
            beta=config.train.beta.value,
            ce_factor=config.model.ce_factor.value,
            score_mode=config.predict.score_mode.value,
            mode=config.predict.mode.value,
            print_every_iter=config.train.print_every_iter.value
        )

    data_module = None
    train_loader = None
    val_loader = None
    if run == "train":
        if config.data.dataset.value == "CuratedImageParameterDataset":
            train_loader = get_dataset(
                base_dir=config.data.data_dir.value,
                num_workers=config.data.num_data_loader_workers.value,
                pin_memory=False,
                batch_size=config.data.batch_size.value,
                mode="train",
                target_size=input_shape[2],
            )
            val_loader = get_dataset(
                base_dir=config.data.data_dir.value,
                num_workers=config.data.num_data_loader_workers.value,
                pin_memory=False,
                batch_size=config.data.batch_size.value,
                mode="val",
                target_size=input_shape[2],
            )
        elif config.data.dataset.value == "SDOMLDatasetV1" or config.data.dataset.value == "SDOMLDatasetV2":
            if config.data.dataset.value == "SDOMLDatasetV1":
                # NOTE due to a bug on Mac, num_workers needs to be 0: https://github.com/pyg-team/pytorch_geometric/issues/366
                data_module = SDOMLv1DataModule(base_dir=config.data.data_dir.value,
                                                num_workers=config.data.num_data_loader_workers.value,
                                                pin_memory=False,
                                                batch_size=config.data.batch_size.value,
                                                channel="171",
                                                target_size=input_shape[2])
            elif config.data.dataset.value == "SDOMLDatasetV2":
                data_module = SDOMLv2DataModule(storage_root=config.data.data_dir.value,
                                                storage_driver=config.data.sdo_ml_v2.storage_driver.value,
                                                num_workers=config.data.num_data_loader_workers.value,
                                                pin_memory=config.data.pin_memory.value,
                                                batch_size=config.data.batch_size.value,
                                                prefetch_factor=config.data.prefetch_factor.value,
                                                channel=config.data.sdo_ml_v2.channel.value,
                                                freq=config.data.sdo_ml_v2.freq.value,
                                                irradiance=config.data.sdo_ml_v2.irradiance.value,
                                                goes_cache_dir=config.data.sdo_ml_v2.goes_cache_dir.value,
                                                target_size=input_shape[2],
                                                train_year=config.data.sdo_ml_v2.train_year.value,
                                                train_start=config.data.sdo_ml_v2.train_start_date.value,
                                                train_end=config.data.sdo_ml_v2.train_end_date.value,
                                                test_year=config.data.sdo_ml_v2.test_year.value,
                                                test_start=config.data.sdo_ml_v2.test_start_date.value,
                                                test_end=config.data.sdo_ml_v2.test_end_date.value,
                                                train_val_split_ratio=config.data.sdo_ml_v2.train_val_split_ratio.value,
                                                train_val_split_temporal_chunk_size=config.data.sdo_ml_v2.train_val_split_temporal_chunk_size.value)

        # config_exclude_keys are already logged from config directly
        wandb_logger = WandbLogger(
            project="sdo-sood", log_model="all", config_exclude_keys=["input_shape",
                                                                      "lr",
                                                                      "z_dim",
                                                                      "model_feature_map_sizes",
                                                                      "use_geco",
                                                                      "beta",
                                                                      "ce_factor",
                                                                      "score_mode",
                                                                      "mode",
                                                                      "print_every_iter"])
        wandb_logger.experiment.config.update(config)

        profiler = None
        if config.train.profile.value == True:
            # https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html
            # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
            from pytorch_lightning.profiler import PyTorchProfiler
            profiler = PyTorchProfiler(
                #  dirpath=work_dir,
                #  filename="output.profile",
                record_shapes=True)

        callbacks = [
            ModelSummary(max_depth=4),
            ModelCheckpoint(monitor="val_loss", dirpath=work_dir / Path("checkpoint"), filename="cevae-{epoch:02d}-{val_loss:.2f}")]
        if config.train.early_stopping.value == True:
            callbacks.append(EarlyStopping(
                monitor="val_loss", mode="min", patience=10))
        trainer = pl.Trainer(logger=wandb_logger,
                             max_epochs=config.train.n_epochs.value,
                             # https://pytorch-lightning.readthedocs.io/en/1.4.7/common/single_gpu.html
                             # distributed training does not yet work because the data loader lambda cannot be pickled
                             gpus=config.devices.gpus.value,
                             profiler=profiler,
                             precision=32,
                             accelerator="auto",
                             default_root_dir=work_dir,
                             callbacks=callbacks)
        # TODO log_graph does not work when running on multiple GPUs
        # # AttributeError: Can't pickle local object 'TorchGraph.create_forward_hook.<locals>.after_forward_hook'
        wandb_logger.watch(cevae_algo, log_graph=False)
        trainer.fit(model=cevae_algo, train_dataloaders=train_loader,
                    val_dataloaders=val_loader, datamodule=data_module)

    if run == "generate":
        cevae_algo.eval()
        cevae_algo.generate()

    if run == "predict":
        cevae_algo.eval()
        pred_dir = config.predict.pred_dir.value
        if pred_dir is None:
            pred_dir = work_dir / Path("predictions")
        elif pred_dir is None and work_dir is None:
            logger.error(
                "Please either provide a log/output dir or a prediction dir")
            sys.exit(0)
        else:
            pred_dir = Path(pred_dir) / Path(current_run_name)

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir, exist_ok=True)

        if config.data.dataset.value == "CuratedImageParameterDataset":
            # TODO use same transforms as during training
            transforms = Compose([Resize((config.model.target_size.value, config.model.target_size.value)),
                                  Grayscale(num_output_channels=1), ToTensor()])
            data_set = ImageFolderWithPaths(config.data.data_dir, transforms)
            data_loader = DataLoader(data_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=1)
        elif config.data.dataset.value == "SDOMLDatasetV1" or config.data.dataset.value == "SDOMLDatasetV2":

            if config.data.dataset.value == "SDOMLDatasetV1":
                # due to a bug on Mac, num processes needs to be 0: https://github.com/pyg-team/pytorch_geometric/issues/366
                data_module = SDOMLv1DataModule(base_dir=config.data.data_dir.value,
                                                num_workers=config.data.num_data_loader_workers.value,
                                                pin_memory=False,
                                                batch_size=1,
                                                channel="171",
                                                target_size=input_shape[2])
            elif config.data.dataset.value == "SDOMLDatasetV2":
                data_module = SDOMLv2DataModule(storage_root=config.data.data_dir.value,
                                                storage_driver=config.data.sdo_ml_v2.storage_driver.value,
                                                num_workers=config.data.num_data_loader_workers.value,
                                                pin_memory=False,
                                                target_size=input_shape[2],
                                                batch_size=1,
                                                prefetch_factor=config.data.prefetch_factor.value,
                                                channel=config.data.sdo_ml_v2.channel.value,
                                                freq=config.data.sdo_ml_v2.freq.value,
                                                irradiance=config.data.sdo_ml_v2.irradiance.value,
                                                goes_cache_dir=config.data.sdo_ml_v2.goes_cache_dir.value,
                                                test_year=config.data.sdo_ml_v2.test_year.value,
                                                test_start=config.data.sdo_ml_v2.test_start_date.value,
                                                test_end=config.data.sdo_ml_v2.test_end_date.value,
                                                skip_train_val=True,
                                                )
                data_loader = data_module.predict_dataloader()
        for index, batch in tqdm(enumerate(data_loader)):
            # TODO attrs are not available for ImageParameterDataset
            img, attrs = batch
            timestamp = parse(attrs["T_OBS"][0])
            wavelength = attrs["WAVELNTH"][0]
            file_name = Path(
                f"{timestamp.strftime(folder_time_format)}_{wavelength}A.png")
            file_path = Path(pred_dir) / file_name
            if config.predict.mode.value == "pixel":
                pixel_scores = cevae_algo.score_pixels(img[0])
                # TODO rather normalize over the full dataset
                save_image(pixel_scores, file_path, normalize=True)
            if config.predict.mode.value == "sample":
                # TODO make it work for batches instead of single images
                sample_score = cevae_algo.score_sample(img[0])
                with open(os.path.join(pred_dir, "predictions.txt"), "a") as target_file:
                    target_file.write(file_path.name + "," +
                                      str(sample_score) + "\n")
