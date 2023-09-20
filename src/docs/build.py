"""Extracts docstrings from source files and builds the documentation."""

import os
import importlib.util


def extract_module_info(module):
    """Extract a module's name, docstring, and path (relative to $PYTHONPATH).
    
    :param module: module to extract info from
    :return: module info
    :rtype: dict
    """
    name = module.__name__
    docstring = module.__doc__
    module_info = {"name": name,
                   "docstring": docstring}
    return module_info


def fetch_module_docs():
    """Walk over the src directory and fetch module info.
    
    :yield: module info
    :rtype: dict
    """
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                module_path = os.path.join(root, file)
                module_name = module_path.rstrip(".py").replace("/", ".")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module_info = extract_module_info(module)
                yield module_info


def build_modules_markdown(path):
    """Build markdown document with module info.
    
    :param str path: path to markdown document
    """
    with open(path, "w") as f:
        f.write("# Modules\n\n")
        for doc in fetch_module_docs():
            print(doc)
            f.write(f"## {doc['name']}\n")
            f.write(f"{doc['docstring']}\n\n")
    return


if __name__ == "__main__":
    build_modules_markdown("docs/modules.md")
