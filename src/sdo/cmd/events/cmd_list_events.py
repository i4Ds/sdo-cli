import os
import click
from sdo.cli import pass_environment
from sdo.events.event_loader import HEKEventManager


@click.command("list", short_help="Lists local events from HEK")
@click.option("--start", type=click.DateTime(), help="Start date")
@click.option("--end", type=click.DateTime(), help="End date")
@click.option("--event-type", type=str, help="Event Type")
@click.option("--db-connection-string", default=lambda: os.environ.get("DB_CONNECTION_STRING", ""))
@pass_environment
def list_events(ctx, start, end, event_type, db_connection_string):
    """Listing cached events from HEK."""
    ctx.log("Listing events...")
    ctx.vlog(
        f"with options: start {start} and end {end} and type {event_type}")

    loader = HEKEventManager(db_connection_string)
    loader.read_events(start, end, event_type)
