import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, relative in {
    'MPLCONFIGDIR': 'runs/matplotlib',
    'IPYTHONDIR': '.jupyter/ipython',
    'JUPYTER_CONFIG_DIR': '.jupyter/config',
    'JUPYTER_RUNTIME_DIR': '.jupyter/runtime',
}.items():
    os.environ[name] = str(ROOT / relative)
os.environ['JUPYTER_PATH'] = str(ROOT / '.jupyter')

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, default=ROOT / 'notebooks/detection_deformable_60min.ipynb')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.notebook
    kernel = ROOT / '.jupyter/kernels/detection-lab'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Detection lab', 'language': 'python',
    }))
    notebook = nbformat.read(args.notebook, as_version=4)

    def progress(cell, cell_index, **kwargs):
        print(f'Executed cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client = NotebookClient(notebook, timeout=900, kernel_name='detection-lab',
                            resources={'metadata': {'path': str(ROOT / 'notebooks')}},
                            on_cell_executed=progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
