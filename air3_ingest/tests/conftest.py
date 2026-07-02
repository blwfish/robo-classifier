"""Shared fixtures for air3_ingest tests."""

import sys
from pathlib import Path

# air3_ingest's own modules import each other bare (e.g. `import merge`,
# `from srt_parser import ...`), matching how they're run in production
# (`python air3_ingest/app.py`, which puts air3_ingest/ itself at sys.path[0]).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
