"""``python -m nanorag.cli`` — the console script's entry point."""

import sys

from nanorag.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
