from sdo.cli import pass_environment
from sdo.sood.algorithms.ae_2d import main

import click


@click.command("train", short_help="Trains an AE model")
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
@pass_environment
def train(ctx,
          target_size=128,
          batch_size=16,
          n_epochs=20,
          lr=1e-4,
          z_dim=128,
          fmap_sizes=(16, 64, 256, 1024),
          print_every_iter=100,
          load_path=None,
          log_dir=None,
          data_dir=None):

    main(run="train",
         target_size=target_size,
         batch_size=batch_size,
         n_epochs=n_epochs,
         lr=lr,
         z_dim=z_dim,
         fmap_sizes=fmap_sizes,
         print_every_iter=print_every_iter,
         load_path=load_path,
         log_dir=log_dir,
         data_dir=data_dir)
