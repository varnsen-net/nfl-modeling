"""Extract docstrings from source files and build the documentation."""

import os
import ast


def fetch_module_docs():
    """Walk over the source directory with ast to fetch module names and
    docstrings.
    
    :return: generator of module names and docstrings
    :rtype: generator
    """
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    tree = ast.parse(f.read())
                docstring = ast.get_docstring(tree)
                full_module_name = path.replace("/", ".")[:-3]
                if docstring:
                    yield {"name": full_module_name,
                           "docstring": docstring}


def build_modules_markdown(path):
    """Build markdown document with module info.
    
    :param str path: path to markdown document
    :return: None
    :rtype: None
    """
    with open(path, "w") as f:
        f.write("# Modules\n\n")
        for doc in fetch_module_docs():
            f.write(f"## {doc['name']}\n")
            f.write(f"{doc['docstring']}\n\n")
    return


if __name__ == "__main__":
    build_modules_markdown("docs/modules.md")
