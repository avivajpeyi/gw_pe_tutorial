"""
Script to execute notebook, build and deploy website
"""

import os
from ploomber_engine import execute_notebook
from nbformat import read, write, NO_CONVERT
import io
import shutil

NOTEBOOK_PATH = "src/workshop_notebook.ipynb"
UNEXECUTED_NOTEBOOK_PATH = "src/unexecuted.ipynb"
KERNEL = "workshop_kernel"


def make_kernel():
    os.system(f"python -m ipykernel install --name '{KERNEL}'")


def _cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell


def strip_output(nb):
    """strip the outputs from a notebook object"""
    nb.metadata.pop('signature', None)
    for cell in _cells(nb):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'prompt_number' in cell:
            cell['prompt_number'] = None
    return nb


def make_unexecuted_notebook():
    with io.open(NOTEBOOK_PATH, 'r', encoding='utf8') as f:
        nb = read(f, as_version=NO_CONVERT)
    nb = strip_output(nb)
    with io.open(UNEXECUTED_NOTEBOOK_PATH, 'w', encoding='utf8') as f:
        write(nb, f)
    with open(UNEXECUTED_NOTEBOOK_PATH, "r") as f:
        txt = f.read()
    txt = txt.replace("Parameter estimation tutorial", "[Unexecuted] Parameter estimation tutorial")
    with open(UNEXECUTED_NOTEBOOK_PATH, "w") as f:
        f.write(txt)



if __name__ == "__main__":
    make_kernel()
    execute_notebook(
        NOTEBOOK_PATH, NOTEBOOK_PATH, log_output=True,
        profile_runtime=True, profile_memory=True, verbose=True
    )
    make_unexecuted_notebook()
    if os.path.exists("_build"):
        shutil.rmtree("_build")
    os.system("jb build src/ --path-output .")
    os.system("ghp-import -n -p -f _build/html")
    print("Done! Check https://avivajpeyi.github.io/nz_gravity_workshop")
