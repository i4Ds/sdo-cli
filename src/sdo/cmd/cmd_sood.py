import click

from sdo.cmd.sood import cmd_ce_vae
from sdo.cmd.sood import cmd_ae


@click.group("sood")
def cli():
    pass


cli.add_command(cmd_ce_vae.cli)
cli.add_command(cmd_ae.cli)
