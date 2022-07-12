from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl

from typing import Any, Optional, Sequence
from pathlib import Path
from dateutil.parser import parse
from torchvision.utils import save_image
import os

folder_time_format = "%Y%m%d-%H%M%S"


class BatchPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, mode: str, save_src_img: bool):
        super().__init__("batch")
        self.output_dir = output_dir
        self.mode = mode
        self.save_src_img = save_src_img

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # TODO attrs are not available for ImageParameterDataset
        _, attrs = batch
        score_path = self.output_dir / Path("predictions.txt")
        if self.mode == "sample" and not os.path.exists(score_path):
            with open(score_path, "a") as target_file:
                target_file.write(
                    f"pixel_pred_path,score,t_obs,wavelength\n")
        for idx, score in enumerate(prediction):
            t_obs = attrs["T_OBS"][idx]
            wavelength = attrs["WAVELNTH"][idx]
            timestamp = parse(t_obs)
            file_name = Path(
                f"{timestamp.strftime(folder_time_format)}_{wavelength}A.png")
            file_path = self.output_dir / file_name
            if self.mode == "pixel":
                # TODO rather normalize over the full dataset
                save_image(score, file_path, normalize=True)
                if self.save_src_img:
                    src_image = batch[0][idx]
                    src_file_name = Path(
                        f"{timestamp.strftime(folder_time_format)}_{wavelength}A_src.png")
                    src_file_path = self.output_dir / src_file_name
                    save_image(src_image, src_file_path, normalize=False)
            elif self.mode == "sample":
                with open(score_path, "a") as target_file:
                    target_file.write(
                        f"{file_path.name},{str(score)},{t_obs},{wavelength}\n")
            else:
                raise ValueError(f"{self.mode} not known")
