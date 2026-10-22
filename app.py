"""MorpHista desktop app: `python app.py` (this is also the file nicegui-pack bundles)."""

if __name__ == "__main__":
    # imported here, not at the top: the native window runs in a second process that re-imports this
    # file, and it must not load torch/CUDA and the models a second time
    from morphista.ui import main

    main()
