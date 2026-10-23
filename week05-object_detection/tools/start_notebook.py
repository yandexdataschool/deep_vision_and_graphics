import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/detection_deformable_60min.ipynb'


def main():
    for name, relative in {
        'JUPYTER_CONFIG_DIR': '.jupyter/config',
        'JUPYTER_DATA_DIR': '.jupyter/data',
        'JUPYTER_RUNTIME_DIR': '.jupyter/runtime',
        'IPYTHONDIR': '.jupyter/ipython',
        'MPLCONFIGDIR': 'runs/matplotlib',
    }.items():
        directory = ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ['JUPYTER_PATH'] = str(ROOT / '.jupyter')
    kernel = ROOT / '.jupyter/kernels/python3'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python — Detection', 'language': 'python',
    }))
    import nbformat
    from nbformat.sign import NotebookNotary
    from notebook.app import main as launch

    NotebookNotary().sign(nbformat.read(NOTEBOOK, as_version=4))
    launch(argv=[str(NOTEBOOK), f'--ServerApp.root_dir={ROOT}', '--ServerApp.ip=127.0.0.1',
                 '--ServerApp.port=8889', '--ServerApp.port_retries=0', *sys.argv[1:]])


if __name__ == '__main__':
    main()
