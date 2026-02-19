# Vision Transformers vs. CNNs for Image Classification

DS 6050 Deep Learning project (UVA). Comparing Vision Transformer and CNN performance on image classification across varying training dataset sizes, using PASCAL VOC (~11K images, 20 classes).

## Overleaf Doc
The link to the group paper draft on Overleaf is here:  [Group 8](https://www.overleaf.com/project/6993a42d2316a7126b7f7ed5).
## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and package management. It's a fast drop-in replacement for `pip` and `venv`.

**Install uv** (one-time):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Set up the environment:**
```bash
uv sync
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and `uv.lock` from the repo, creates the `.venv` if it doesn't exist, and installs exactly the right packages. When someone adds a new dependency, just pull and run `uv sync` again.

**Adding new packages:**
```bash
uv add torch torchvision   # adds to pyproject.toml and installs
```

This keeps dependencies tracked in the repo so everyone stays in sync. It's also much faster than `pip` — installs that take minutes with pip usually finish in seconds with uv. You can also use `uv pip install` as a drop-in for `pip` if you just want to install something without tracking it.

## Repository Structure

- `literature/` — Papers, citations, and literature review report
- `resources/` — Course materials (dataset info, project slides)
- `plans/` — Project plans and implementation logs
