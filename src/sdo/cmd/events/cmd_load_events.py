import os
import click
from sdo.cli import pass_environment
from sdo.events.event_loader import HEKEventManager


@click.command("get", short_help="Loads events from HEK")
@click.option("--start", default='2012-12-01T01:00:00', type=click.DateTime(), help="Start date")
@click.option("--end", default='2012-12-03T00:00:00', type=click.DateTime(), help="End date")
@click.option("--event-type", default='AR', type=str, help="Event Type")
@click.option("--db-connection-string", default=lambda: os.environ.get("DB_CONNECTION_STRING", ""))
@pass_environment
def load_events(ctx, start, end, event_type, db_connection_string):
    """Loads events from HEK."""
    ctx.log("Starting to load events...")
    ctx.vlog(
        f"with options: start {start} and end {end} and type {event_type}")

    loader = HEKEventManager(db_connection_string)
    loader.load_events_from_hek(start, end, event_type)
