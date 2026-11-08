"""Allow `python -m airjam` from anywhere."""

import sys

from .app import main

sys.exit(main())
