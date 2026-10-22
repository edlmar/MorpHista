"""MorpHista desktop app: `python app.py` (this is also the file nicegui-pack bundles)."""

import sys
from pathlib import Path

if sys.stdout is None:  # started without a console (the desktop shortcut runs pythonw): write a log instead
    sys.stdout = sys.stderr = open(Path(__file__).with_name("morphista.log"), "a", encoding="utf-8", buffering=1)

if __name__ == "__main__":
    # imported here, not at the top: the native window runs in a second process that re-imports this
    # file, and it must not load torch/CUDA and the models a second time
    from morphista.ui import main

    main()
