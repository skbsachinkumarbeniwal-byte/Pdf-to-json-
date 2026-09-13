import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import pymupdf BEFORE any per-test capture exists: pymupdf binds its
# message stream (sys.stdout) at import time, and a per-test capsys
# stream is closed when that test ends -> later pymupdf messages (e.g.
# the fitz-deprecation note) would raise "I/O operation on closed file".
import pymupdf  # noqa: E402,F401
