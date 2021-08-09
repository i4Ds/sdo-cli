import os
import click
from sdo.cli import pass_environment
from sdo.events.bboxes import compute_ious


@click.command("analyze", short_help="Analyzes model outputs and compares it with events from HEK")
@click.option("--src-dir", default="./notebooks/data/aia_171_bounding", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--sood-dir", default="./notebooks/output/aia_171_bounding_pred/20210626-153237_cevae/predictions", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--out-dir", default="./output/analytics", type=click.Path(file_okay=False, resolve_path=True))
@click.option("--event-type", default=["AR", "FL"], multiple=True, type=str, help="HEK Event Type")
@click.option("--db-connection-string", default=lambda: os.environ.get("DB_CONNECTION_STRING", ""))
@pass_environment
def analyze_events(ctx, src_dir, sood_dir, out_dir, event_type, db_connection_string):
    """Loads events from HEK."""
    ctx.log("Starting to analyze events...")
    ctx.vlog(
        f"with options: src_dir {src_dir}, sood_dir {sood_dir}, out_dir {out_dir} and type {event_type}")
    compute_ious(src_dir, sood_dir, out_dir,
                 db_connection_string, hek_event_types=event_type)
