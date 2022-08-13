# adjusted from https://github.com/MIC-DKFZ/mood, Credit: D. Zimmerer
# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
import json
import datetime
import os
import sys
from pathlib import Path
from typing import Any
import logging
from dateutil.parser import parse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as dist
import wandb
from torch.nn import MSELoss, L1Loss
from torchmetrics import StructuralSimilarityIndexMeasure
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary)
from pytorch_lightning.loggers import WandbLogger
from sdo.sood.data.image_dataset import get_dataset
from sdo.sood.data.path_dataset import ImageFolderWithPaths
from sdo.sood.data.sdo_ml_v1_dataset import SDOMLv1DataModule
from sdo.sood.data.sdo_ml_v2_dataset import SDOMLv2DataModule
from sdo.sood.models.aes import VAE
from sdo.sood.util.ce_noise import get_square_mask, normalize, smooth_tensor
from sdo.sood.util.prediction_writer import BatchPredictionWriter
from sdo.sood.util.utils import (get_smooth_image_gradient, read_config,
                                 tensor_to_image, save_image_grid)
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from torchvision.utils import save_image
import time


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
        print_every_iter,
        debug: bool,
        work_dir
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
        self.debug = debug
        self.work_dir = work_dir

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
        t_start = time.process_time()

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
                # TODO verify the impact of sampling
                x_rec_ce, _ = self.model(inpt_noisy, sample=False)
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

        elapsed_time = time.process_time() - t_start
        self.log('elapsed_time', elapsed_time)

        return {"loss": loss, "loss_vae": loss_vae, "loss_ce": loss_ce, "loss_vae_kl": loss_vae_kl, "loss_vae_rec": loss_vae_rec}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        # TODO why no sampling during validation?
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
        return {"loss": loss, "val_loss_kl": kl_loss, "val_loss_rec": rec_loss}

    def reconstruct(self, batch, sample=False):
        x, attrs = batch
        save_path = self.work_dir / Path("reconstructions")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger.info(f"writing reconstructions to {save_path}")

        ssim = StructuralSimilarityIndexMeasure()
        mse = MSELoss()
        mae = L1Loss()
        loss_path = save_path / Path("reconstruction_loss.csv")
        if not os.path.exists(loss_path):
            with open(loss_path, "a") as target_file:
                target_file.write(
                    f"pixel_pred_path,ssim,mse,mae,t_obs,wavelength\n")
        with torch.no_grad():
            mu, std = self.model.encode(x)
            z_dist = dist.Normal(mu, std)
            if sample:
                z_sample = z_dist.rsample()
            else:
                z_sample = mu
            x_rec = self.model.decode(z_sample)

            for idx in range(len(x_rec)):
                x_i, x_rec_i, mu_i, std_i = x[idx], x_rec[idx], mu[idx], std[idx]

                t_obs = attrs["T_OBS"][idx]
                wavelength = attrs["WAVELNTH"][idx]
                timestamp = parse(t_obs)
                file_name = f"{timestamp.strftime(folder_time_format)}_{wavelength}A"
                input_img = tensor_to_image(x[idx])
                input_img.save(save_path / Path(
                    f"{file_name}_src.png"))
                rec_img = tensor_to_image(x_rec_i)
                rec_img.save(save_path / Path(
                    f"{file_name}_rec.png"))

                torch.save(mu_i, save_path / Path(
                    f"{file_name}_mu.pt"))
                torch.save(std_i, save_path / Path(
                    f"{file_name}_std.pt"))

                x_rec_i = x_rec_i[None, :, :, :]
                x_i = x_i[None, :, :, :]
                x_ssim = ssim(x_rec_i, x_i)
                x_mse = mse(x_rec_i, x_i)
                x_mae = mae(x_rec_i, x_i)
                with open(loss_path, "a") as target_file:
                    target_file.write(
                        f"{file_name},{str(x_ssim.numpy())},{str(x_mse.numpy())},{str(x_mae.numpy())},{t_obs},{wavelength}\n")

    def test_step(self, batch, batch_idx):
        logger.warn("test step not defined")
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch, batch_idx=batch_idx)

    def forward(self, batch, batch_idx: int = None):
        if self.mode == "sample":
            return self.score_sample(batch)
        elif self.mode == "pixel":
            return self.score_pixels(batch, batch_idx=batch_idx)
        else:
            raise ValueError(f"invalid mode {self.mode}")

    def generate(self, n_samples=1, n_iter=128, mu=None, std=None, with_cm=False):
        output_dir = Path(self.work_dir) / Path("generated")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info(f"writing generated examples to {output_dir}")
        for i in range(n_iter):
            if mu is None:
                mu = torch.zeros_like(torch.empty(self.z_dim, 1, 1))
            if std is None:
                std = torch.ones_like(torch.empty(self.z_dim, 1, 1))

            p = torch.distributions.Normal(mu, std)
            z = p.rsample((n_samples,))

            with torch.no_grad():
                pred = self.model.decode(z.to(self.device)).cpu()

            file_name = output_dir / \
                f"{datetime.datetime.now().strftime(folder_time_format)}_{i}_generated.jpeg"

            if with_cm:
                from PIL import Image
                from sunpy.visualization.colormaps import cm
                from torchvision.utils import make_grid

                grid = make_grid(pred, normalize=True)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
                    1, 2, 0).to("cpu", torch.uint8).numpy()
                m = cm.cmlist.get('sdoaia%d' % int(171))
                v = np.squeeze(ndarr[:, :, 0])
                v = m(v)
                v = (v[:, :, :3]*255).astype(np.uint8)
                im = Image.fromarray(v)
                im.save(file_name)
            else:
                save_image(pred, file_name, normalize=True, scale_each=True)

    def score_sample(self, batch):
        data, _ = batch
        x_rec, z_dist = self.model(data, sample=False)
        kl_loss = self.kl_loss_fn(z_dist, sum_samples=False)
        rec_loss = self.rec_loss_fn(x_rec, data, sum_samples=False)
        sample_scores = kl_loss * self.beta + rec_loss * self.theta

        return sample_scores.cpu().tolist()

    def score_pixels(self, batch, batch_idx: int = None):
        data, _ = batch
        with torch.enable_grad():
            x_rec, _ = self.model(data, sample=False)
            if self.score_mode == "combi":
                rec = torch.pow((x_rec - data), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)

                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    loss = self.kl_loss_fn(z_d)
                    return loss
                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=data, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(
                    loss_grad_kl, dim=1, keepdim=True)
                pixel_scores = smooth_tensor(
                    normalize(loss_grad_kl), kernel_size=8) * rec
            elif self.score_mode == "rec":
                rec = torch.pow((x_rec - data), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)
                pixel_scores = rec
            elif self.score_mode == "kl_grad":
                rec = torch.pow((x_rec - data), 2).detach().cpu()
                rec = torch.mean(rec, dim=1, keepdim=True)

                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    loss = self.kl_loss_fn(z_d)
                    return loss
                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=data, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(
                    loss_grad_kl, dim=1, keepdim=True)
                pixel_scores = smooth_tensor(
                    normalize(loss_grad_kl), kernel_size=8)
            elif self.score_mode == "elbo_grad":
                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    kl_loss_ = self.kl_loss_fn(z_d)
                    rec_loss_ = self.rec_loss_fn(x_r, x)
                    loss_ = kl_loss_ * self.beta + rec_loss_ * self.theta
                    return torch.mean(loss_)
                loss_grad_kl = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=data, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_kl = torch.mean(
                    loss_grad_kl, dim=1, keepdim=True)
                pixel_scores = smooth_tensor(
                    normalize(loss_grad_kl), kernel_size=8)
            elif self.score_mode == "rec_grad":
                def __err_fn(x):
                    x_r, z_d = self.model(x, sample=False)
                    loss = self.rec_loss_fn(x_r, x)
                    return torch.mean(loss)
                loss_grad_rec = (
                    get_smooth_image_gradient(
                        model=self.model, inpt=data, err_fn=__err_fn, grad_type="vanilla", n_runs=2
                    )
                    .detach()
                    .cpu()
                )
                loss_grad_rec = torch.mean(
                    loss_grad_rec, dim=1, keepdim=True)
                pixel_scores = smooth_tensor(
                    normalize(loss_grad_rec), kernel_size=8)
            pixel_scores = pixel_scores.detach().cpu()
            if self.debug:
                logger.info(
                    f"Writing debug information about predictions to {self.work_dir}")
                save_image_grid(data, name="Input", save_dir=self.work_dir, image_args={
                    "normalize": True}, n_iter=batch_idx)
                save_image_grid(x_rec, name="Reconstruction", save_dir=self.work_dir, image_args={
                    "normalize": True}, n_iter=batch_idx)
                save_image_grid(pixel_scores, name="Scores", save_dir=self.work_dir, image_args={
                    "normalize": True}, n_iter=batch_idx)

        return pixel_scores

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


folder_time_format = "%Y%m%d-%H%M%S"


def main(
    run: str = "train",
    config_file: Path = Path("./config/defaults.yaml"),
    predict_mode: str = None,
    config_overrides: dict = None
):
    config = read_config(config_file, config_overrides)
    logger.info("found config")
    logger.info(json.dumps(config, indent=2))

    predict_mode = predict_mode or config.predict.mode.value

    current_run_name = f"{datetime.datetime.now().strftime(folder_time_format)}_cevae"
    work_dir = Path(config.log_dir.value) / Path(current_run_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    input_shape = (config.data.batch_size.value, 1,
                   config.model.target_size.value, config.model.target_size.value)

    if config.model.load_path.value is not None:
        cevae_algo = ceVAE.load_from_checkpoint(
            config.model.load_path.value, mode=predict_mode,  debug=config.debug.value, work_dir=work_dir, score_mode=config.predict.score_mode.value)
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
            mode=predict_mode,
            print_every_iter=config.train.print_every_iter.value,
            debug=config.debug.value,
            work_dir=work_dir
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
                                                train_val_split_temporal_chunk_size=config.data.sdo_ml_v2.train_val_split_temporal_chunk_size.value,
                                                sampling_strategy=config.data.sdo_ml_v2.sampling_strategy.value,
                                                mask_limb=config.data.sdo_ml_v2.mask_limb.value,
                                                mask_limb_radius_scale_factor=config.data.sdo_ml_v2.mask_limb_radius_scale_factor.value,
                                                reduce_memory=config.data.sdo_ml_v2.reduce_memory)

        # config_exclude_keys are already logged from config directly
        wandb_logger = WandbLogger(
            project="sdo-sood",
            log_model="all",
            id=config.train.wandb_run_id.value,
            config_exclude_keys=["input_shape",
                                 "lr",
                                 "z_dim",
                                 "model_feature_map_sizes",
                                 "use_geco",
                                 "beta",
                                 "ce_factor",
                                 "score_mode",
                                 "mode",
                                 "print_every_iter"])
        wandb_logger.experiment.config.update(config, allow_val_change=True)

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
                             fast_dev_run=config.train.fast_dev_run.value,
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

    if run == "predict" or run == "reconstruct":
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
                                                obs_times=config.data.sdo_ml_v2.obs_times.value,
                                                target_size=input_shape[2],
                                                batch_size=config.data.batch_size.value,
                                                prefetch_factor=config.data.prefetch_factor.value,
                                                channel=config.data.sdo_ml_v2.channel.value,
                                                freq=config.data.sdo_ml_v2.freq.value,
                                                irradiance=config.data.sdo_ml_v2.irradiance.value,
                                                goes_cache_dir=config.data.sdo_ml_v2.goes_cache_dir.value,
                                                test_year=config.data.sdo_ml_v2.test_year.value,
                                                test_start=config.data.sdo_ml_v2.test_start_date.value,
                                                test_end=config.data.sdo_ml_v2.test_end_date.value,
                                                skip_train_val=True,
                                                mask_limb=config.data.sdo_ml_v2.mask_limb.value,
                                                mask_limb_radius_scale_factor=config.data.sdo_ml_v2.mask_limb_radius_scale_factor.value,
                                                reduce_memory=config.data.sdo_ml_v2.reduce_memory)
                data_loader = data_module.predict_dataloader()
        if run == "reconstruct":
            for _, batch in enumerate(data_module.predict_dataloader()):
                cevae_algo.reconstruct(batch)
        elif run == "predict":
            logger.info(f"logging predictions to {pred_dir}")
            trainer = pl.Trainer(
                gpus=config.devices.gpus.value, accelerator="auto", callbacks=[BatchPredictionWriter(output_dir=pred_dir, mode=predict_mode, save_src_img=config.predict.save_src_img.value)])
            trainer.predict(cevae_algo, data_loader, return_predictions=False)
