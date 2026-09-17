import sys
from pathlib import Path

# Ensures `import src...` and `import scripts...` work when pytest is run
# from any working directory (e.g. `pytest`, `pytest tests/`, or from an IDE).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
