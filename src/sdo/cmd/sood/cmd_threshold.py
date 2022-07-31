import click

from sdo.cmd.sood.threshold import cmd_predict


@click.group("threshold")
def cli():
    pass


cli.add_command(cmd_predict.predict)
