import os
import sys

import click

from ivory import __version__

pgk_dir = os.path.dirname(os.path.abspath(__file__))
version_msg = f"{__version__} from {pgk_dir} (Python {sys.version[:3]})."


@click.command()
@click.version_option(version_msg, "-V", "--version")
@click.argument("path", nargs=-1, type=click.Path(exists=True))
def cli(path):
    pass
