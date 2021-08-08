import click

from sdo.cmd.goes import cmd_download


@click.group("goes")
def cli():
    pass


cli.add_command(cmd_download.download)
