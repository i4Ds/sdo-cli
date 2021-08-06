import click

from sdo.cmd.events import cmd_load_events
from sdo.cmd.events import cmd_list_events
from sdo.cmd.events import cmd_analyze_events


@click.group("events")
def cli():
    pass


cli.add_command(cmd_load_events.load_events)
cli.add_command(cmd_list_events.list_events)
cli.add_command(cmd_analyze_events.analyze_events)
