# Contributing

Thanks for your interest in improving the HDF5 Viewer — SEXTANTS Edition!

## Getting set up

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
python main.py
```

Python 3.11+ is required.

## Before opening a pull request

Run the test suite and the formatters/linters:

```bash
pytest                                       # headless Qt tests
isort . && black . && flake8 . && mypy .     # formatting & static checks
```

Continuous integration runs the same checks on every push and PR
(see [.github/workflows/ci.yaml](.github/workflows/ci.yaml)); please make sure they pass
locally first.

## Adding a new analysis tool

The numerical cores live in [src/recon/](src/recon/), independent of the GUI, and the shared
tool interface (how a tool plugs into the main window, receives datasets, and where the math
goes) is documented in [docs/CREATING_TOOLS.md](docs/CREATING_TOOLS.md), with a copy-paste
template. New reconstruction logic should come with unit tests in [tests/](tests/).

## Guidelines

- Keep GUI code and numerical/reconstruction code separated so the math stays testable.
- Match the style of the surrounding code; let `black`/`isort` handle formatting.
- Bump the version in [src/version.py](src/version.py) (Semantic Versioning) when releasing.

## License

This project is licensed under the **GNU General Public License v3** (see [LICENSE](LICENSE)).
By contributing, you agree that your contributions will be licensed under the same terms.
