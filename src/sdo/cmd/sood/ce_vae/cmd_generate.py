from sdo.cli import pass_environment
from sdo.sood.algorithms.ce_vae import main

import click


@click.command("generate", short_help="Generate a set of images with the CE-VAE model (requires a pretrained model)")
@click.option("-m", "--mode", default="pixel", type=click.Choice(["pixel", "sample"], case_sensitive=False))
@click.option("--target-size", type=click.IntRange(1, 512, clamp=True), default=128)
@click.option("--batch-size", type=click.IntRange(1, 512, clamp=True), default=16)
@click.option("--n-epochs", type=int, default=20)
@click.option("--lr", type=float, default=1e-4)
@click.option("--z-dim", type=int, default=128)
@click.option("-fm", "--fmap-sizes", type=int, multiple=True, default=[16, 64, 256, 1024])
@click.option("--print-every-iter", type=int, default=100)
@click.option("-l", "--load-path", type=click.Path(exists=True), required=False, default=None)
@click.option("-o", "--log-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option(
    "--logger", type=click.Choice(["visdom", "tensorboard", "file"], case_sensitive=False), required=False, default="visdom"
)
@click.option("-t", "--test-dir", type=click.Path(exists=True), required=False, default=None)
@click.option("-p", "--pred-dir", type=click.Path(exists=True, writable=True), required=False, default=None)
@click.option("-d", "--data-dir", type=click.Path(exists=True), required=True, default=None)
@click.option("--use-geco", type=bool, is_flag=True, default=False)
@click.option("--beta", type=float, default=0.01)
@click.option("--ce-factor", type=click.FloatRange(0.0, 1.0, clamp=True), default=0.5)
@click.option(
    "--score-mode", type=click.Choice(["rec", "grad", "combi"], case_sensitive=False), required=False, default="rec"
)
@pass_environment
def generate(ctx,
             mode,
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
             logger,
             test_dir,
             pred_dir,
             data_dir):

    main(mode=mode,
         run="generate",
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
         logger=logger,
         test_dir=test_dir,
         pred_dir=pred_dir,
         data_dir=data_dir)
