import click

from sdo.cmd.sood.ce_vae import cmd_train, cmd_generate, cmd_predict, cmd_reconstruct


@click.group("ce_vae")
def cli():
    pass


cli.add_command(cmd_train.train)
cli.add_command(cmd_generate.generate)
cli.add_command(cmd_predict.predict)
cli.add_command(cmd_reconstruct.reconstruct)
