import click

from sdo.cmd.data import cmd_download
from sdo.cmd.data import cmd_patch
from sdo.cmd.data import cmd_resize


@click.group("data")
def cli():
    pass


cli.add_command(cmd_download.download)
cli.add_command(cmd_patch.patch)
cli.add_command(cmd_resize.resize)
