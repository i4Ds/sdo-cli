from sdo.cli import pass_environment
from sdo.sood.algorithms.ce_vae import main

import click


@click.command("train", short_help="Trains a CE-VAE model")
@click.option("--target-size", type=click.IntRange(1, 512, clamp=True), default=128)
@click.option("--batch-size", type=click.IntRange(1, 512, clamp=True), default=16)
@click.option("--n-epochs", type=int, default=20)
@click.option("--lr", type=float, default=1e-4)
@click.option("--z-dim", type=int, default=128)
@click.option("-fm", "--fmap-sizes", type=int, multiple=True, default=[16, 64, 256, 1024])
@click.option("--print-every-iter", type=int, default=100)
@click.option("-l", "--load-path", type=click.Path(exists=True), required=False, default=None)
@click.option("-o", "--log-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option("-d", "--data-dir", type=click.Path(exists=True), required=True, default=None)
@click.option("--use-geco", type=bool, is_flag=True, default=False)
@click.option("--beta", type=float, default=0.01)
@click.option("--ce-factor", type=click.FloatRange(0.0, 1.0, clamp=True), default=0.5)
@click.option(
    "--score-mode", type=click.Choice(["rec", "grad", "combi"], case_sensitive=False), required=False, default="rec"
)
@click.option(
    "--dataset", type=click.Choice(["CuratedImageParameterDataset", "SDOMLDatasetV1", "SDOMLDatasetV2"], case_sensitive=False), required=False, default="CuratedImageParameterDataset"
)
@click.option("--num-data-loader-workers", type=int, default=0)
@click.option("--train-start-date", type=str, default="2010-01-01 00:00:00")
@click.option("--train-end-date", type=str, default="2010-08-30 23:59:59")
@click.option("--test-start-date", type=str, default="2010-09-01 00:00:00")
@click.option("--test-end-date", type=str, default="2010-12-31 23:59:59")
@click.option("--train-val-split-ratio", type=float, default=0.8)
@click.option("--train-val-split-temporal-chunk-size", type=str, default="14d")
@pass_environment
def train(ctx,
          target_size,
          batch_size,
          n_epochs,
          lr,
          z_dim,
          fmap_sizes,
          use_geco,
          beta,
          ce_factor,
          score_mode,
          print_every_iter,
          load_path,
          log_dir,
          data_dir,
          dataset,
          num_data_loader_workers,
          train_start_date,
          train_end_date,
          test_start_date,
          test_end_date,
          train_val_split_ratio,
          train_val_split_temporal_chunk_size):

    main(run="train",
         target_size=target_size,
         batch_size=batch_size, n_epochs=n_epochs,
         lr=lr,
         z_dim=z_dim,
         fmap_sizes=fmap_sizes,
         use_geco=use_geco,
         beta=beta,
         ce_factor=ce_factor,
         score_mode=score_mode,
         print_every_iter=print_every_iter,
         load_path=load_path,
         log_dir=log_dir,
         data_dir=data_dir,
         dataset=dataset,
         num_data_loader_workers=num_data_loader_workers,
         train_start=train_start_date,
         train_end=train_end_date,
         test_start=test_start_date,
         test_end=test_end_date,
         train_val_split_ratio=train_val_split_ratio,
         train_val_split_temporal_chunk_size=train_val_split_temporal_chunk_size
         )
