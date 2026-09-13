"""Execute and save the pilot notebook with its own project Python kernel."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_PATH"] = str(root / ".venv-project/share/jupyter")
os.environ["JUPYTER_RUNTIME_DIR"] = str(root / ".venv-project/jupyter-runtime")
path = root / "notebooks/01_f1_project_lab.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=600, kernel_name="mma3001", resources={"metadata":{"path":str(root)}}).execute()
nbformat.write(nb, path)
print("Executed", sum(c.cell_type == "code" for c in nb.cells), "code cells successfully")
