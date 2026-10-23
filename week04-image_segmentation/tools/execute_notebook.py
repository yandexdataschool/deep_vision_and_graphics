import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter/ipython"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".jupyter/config"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter/runtime"))
os.environ["JUPYTER_PATH"] = str(ROOT / ".jupyter")

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/segmentation_transformers.ipynb")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/segmentation_transformers.ipynb")
    args = parser.parse_args()
    kernel = ROOT / ".jupyter/kernels/segmentation-lab"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Segmentation lab", "language": "python",
    }))
    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    workdir = args.workdir.resolve() if args.workdir else path.parent
    client = NotebookClient(notebook, timeout=300, kernel_name="segmentation-lab",
                            resources={"metadata": {"path": str(workdir)}})
    client.execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {args.output}")


if __name__ == "__main__":
    main()
