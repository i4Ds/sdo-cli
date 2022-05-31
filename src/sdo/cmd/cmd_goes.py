import click

from sdo.cmd.goes import cmd_download, cmd_get


@click.group("goes")
def cli():
    pass


cli.add_command(cmd_download.download)
cli.add_command(cmd_get.get)
