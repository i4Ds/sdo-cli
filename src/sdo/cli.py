import os

import click
import traceback
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-4s %(message)s')
logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(auto_envvar_prefix="SDO")


class Environment:
    def __init__(self):
        self.verbose = False
        self.home = os.getcwd()

    def log(self, msg, *args):
        if args:
            msg %= args
        logger.info(msg)

    def vlog(self, msg, *args):
        if args:
            msg %= args
        logger.debug(msg)


pass_environment = click.make_pass_decorator(Environment, ensure=True)
cmd_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "cmd"))


class SolarDynamicsObservatoryCLI(click.MultiCommand):
    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(cmd_folder):
            if filename.endswith(".py") and filename.startswith("cmd_"):
                rv.append(filename[4:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        try:
            mod = __import__(
                f"sdo.cmd.cmd_{name}", None, None, ["cli"])
        except ImportError as e:
            print(e)
            traceback.print_exc()
            return
        return mod.cli


@click.command(cls=SolarDynamicsObservatoryCLI, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--home",
    type=click.Path(exists=False, file_okay=False, resolve_path=True),
    default="./.sdo-cli",
    help="Changes the folder to operate on.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enables verbose mode.")
@pass_environment
def cli(ctx, verbose, home):
    """CLI to manipulate and model SDO data."""
    ctx.verbose = verbose
    if verbose:
        logger.setLevel(logging.DEBUG)

    if home is not None:
        ctx.home = home

    if os.path.exists(home) and not os.path.isdir(home):
        raise ValueError(home + " is not a directory")

    if not os.path.exists(home):
        os.makedirs(home)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
