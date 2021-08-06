import click

from sdo.cmd.sood.ae import cmd_train, cmd_predict


@click.group("ae")
def cli():
    pass


cli.add_command(cmd_train.train)
cli.add_command(cmd_predict.predict)
