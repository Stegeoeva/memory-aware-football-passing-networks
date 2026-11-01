"""Execute and save the notebook using this exact Python interpreter."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / ".runtime"
runtime.mkdir(exist_ok=True)
for name in ["JUPYTER_RUNTIME_DIR", "IPYTHONDIR", "MPLCONFIGDIR"]:
    os.environ[name] = str(runtime / name.lower())
    Path(os.environ[name]).mkdir(exist_ok=True)
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="football_passing_networks.ipynb")
    args = parser.parse_args()
    path = (ROOT / "notebooks" / args.notebook).resolve()
    if path.parent != ROOT / "notebooks" or path.suffix != ".ipynb":
        raise ValueError("Select a notebook file in the project's notebooks directory")
    nb = nbformat.read(path,as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, timeout=300, resources={"metadata":{"path":str(ROOT)}}, km=manager)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(nb,path)
    errors = [o for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs if o.output_type == "error"]
    assert not errors, errors
    count = sum(cell.cell_type == "code" and cell.execution_count is not None for cell in nb.cells)
    images = sum("image/png" in o.get("data",{}) for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs)
    print(f"Executed {count} code cells with {images} embedded figures and no errors: {path}")
