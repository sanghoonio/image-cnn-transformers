# Vision Transformers vs. CNNs for Image Classification

DS 6050 Deep Learning project (UVA). Comparing Vision Transformer and CNN performance on image classification across varying training dataset sizes, using PASCAL VOC (~11K images, 20 classes).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and package management. It's a fast drop-in replacement for `pip` and `venv`.

**Install uv** (one-time):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create and activate the virtual environment:**
```bash
uv venv .venv
source .venv/bin/activate
```

**Install all project dependencies:**
```bash
uv sync
```

That's it. `uv sync` reads the `pyproject.toml` lockfile in the repo, creates the venv if needed, and installs exactly the right packages. When someone adds a new dependency, just pull and run `uv sync` again.

**Adding new packages:**
```bash
uv add torch torchvision   # adds to pyproject.toml and installs
```

This keeps dependencies tracked in the repo so everyone stays in sync. It's also much faster than `pip` — installs that take minutes with pip usually finish in seconds with uv. You can also use `uv pip install` as a drop-in for `pip` if you just want to install something without tracking it.

## Repository Structure

- `literature/` — Papers, citations, and literature review report
- `plans/` — Project plans and implementation logs
