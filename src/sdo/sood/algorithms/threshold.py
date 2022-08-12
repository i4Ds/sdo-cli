# adjusted from https://github.com/MIC-DKFZ/mood, Credit: D. Zimmerer
# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
import json
import datetime
import os
import sys
from pathlib import Path
import logging
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from dateutil.parser import parse
from sdo.sood.util.utils import read_config
from PIL import Image
from sdo.sood.data.sdo_ml_v2_dataset import SDOMLv2DataModule
logger = logging.getLogger(__name__)

folder_time_format = "%Y%m%d-%H%M%S"


def main(
    run: str = "predict",
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

    data_module = None

    if run == "predict":
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

        if config.data.dataset.value == "SDOMLDatasetV2":
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
        logger.info(f"logging predictions to {pred_dir}")
        for batch_idx, samples in enumerate(data_module.predict_dataloader()):
            X, y = samples
            # TODO over the full dataset?
            for x, t_obs, wavelength in zip(X, y["T_OBS"], y["WAVELNTH"]):
                timestamp = parse(t_obs)
                file_name = Path(
                    f"{timestamp.strftime(folder_time_format)}_{wavelength}A.png")
                file_path = pred_dir / file_name
                grid = make_grid(x, normalize=True)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
                    1, 2, 0).to("cpu", torch.uint8).numpy()
                threshold = ndarr.mean() + 2.0 * ndarr.std()

                if predict_mode == "sample":
                    score_path = pred_dir / Path("predictions.txt")

                    if not os.path.exists(score_path):
                        with open(score_path, "a") as target_file:
                            target_file.write(
                                f"pixel_pred_path,score,t_obs,wavelength\n")
                    score = x.mean().numpy()
                    with open(score_path, "a") as target_file:
                        target_file.write(
                            f"{file_path.name},{str(score)},{t_obs},{wavelength}\n")
                elif predict_mode == "pixel":
                    ndarr[ndarr <= threshold] = 0

                    def normalize(img, low, high):
                        img = img.clip(low, high)
                        img = (img - low) / max(high - low, 1e-5)
                        return img
                    ndarr = normalize(ndarr, ndarr.min(), ndarr.max())
                    ndarr = (ndarr[:, :, :]*255).astype(np.uint8)
                    img = Image.fromarray(ndarr)
                    img.save(file_path)
                    if config.predict.save_src_img.value:
                        src_image = x
                        src_file_name = Path(
                            f"{timestamp.strftime(folder_time_format)}_{wavelength}A_src.png")
                        src_file_path = pred_dir / src_file_name
                        save_image(src_image, src_file_path, normalize=False)
                else:
                    raise ValueError(f"invalid mode {predict_mode}")
