"""This Sphinx extension automagically builds a simple set of API documentation pages."""

import os
import pathlib

from conf import source_code_dir, api_source_docs_dir


index_template = """
API Reference
=============

.. toctree::
   
"""


module_template = """
{module_name}
{adornment}

.. automodule:: {module_name}
   :members:
   :show-inheritance:
"""


class ApiBuilder:
    """Builds the API documentation pages.
    
    :param pathlib.Path _cwd: The current working directory.
    :param str _source_code_path: The path to the source code directory.
    :param str _api_source_docs_path: The path to the API source docs directory.
    :param str _index_template: The template for the API index page.
    """


    _cwd = pathlib.Path.cwd()
    _source_code_path = _cwd.parent / source_code_dir
    _api_source_docs_path = _cwd / 'source' / api_source_docs_dir
    _index_template = index_template
    _module_template = module_template


    def format_file_name(self, file, root, starting_idx):
        """Formats a file name + root path from os.walk into a module name
        in the style of a Python import statement, e.g. 'src.data.utils'.
    
        :param str file: The file name, with extension.
        :param str root: The root path of the file.
        :param int starting_idx: The index of the root path parts to start from.
        :return: The module name in the style of a Python import statement.
        :rtype: str
        """
        file = file[:-3]
        file_path = pathlib.Path(root) / file
        relative_parts = file_path.parts[starting_idx:]
        module_name = '.'.join(relative_parts)
        module_name = f"{source_code_dir}.{module_name}"
        return module_name


    def remove_element(self, ele, l):
        """Remove an element from a list if it exists.
        
        :param str ele: The element to remove.
        :param list l: The list to remove the element from.
        :return: The list with the element removed.
        :rtype: list
        """
        if ele in l:
            l.remove(ele)
        return l


    def get_source_code_modules(self):
        """Get the modules from the source code path.
        
        :yield: The modules from the source code path.
        :rtype: str
        """
        starting_idx = len(self._source_code_path.parts)
        for root, dirs, files in os.walk(self._source_code_path):
            self.remove_element('__pycache__', dirs)
            self.remove_element('__init__.py', files)
            for file in files:
                if file.endswith('.py'):
                    yield self.format_file_name(file, root, starting_idx)
    
    def build_index_page(self, sorted_modules):
        """Build the index page for the API documentation and write to disk.
        
        :param list sorted_modules: The sorted list of modules.
        :return: None
        :rtype: None
        """
        index_page = self._index_template
        for m in sorted_modules:
            index_page = index_page + f"   {m}\n"
        with open(self._api_source_docs_path / 'index.rst', 'w') as f:
            f.write(index_page)
        return


    def build_module_pages(self, sorted_modules):
        """Build the module pages for the API documentation and write to disk.
        
        :param list sorted_modules: The sorted list of modules.
        :return: None
        :rtype: None
        """
        for m in sorted_modules:
            adornment = '=' * len(m)
            module_page = self._module_template.format(module_name=m,
                                                       adornment=adornment)
            with open(self._api_source_docs_path / f'{m}.rst', 'w') as f:
                f.write(module_page)
        return


    def run(self):
        """Run the extension.
        
        :return: None
        :rtype: None
        """
        os.makedirs(self._api_source_docs_path, exist_ok=True)
        modules = [m for m in self.get_source_code_modules()]
        sorted_modules = sorted(modules)
        self.build_index_page(sorted_modules)
        self.build_module_pages(sorted_modules)
        return


def setup(app):
    """Set up the Sphinx extension.

    :param app: The Sphinx application object.
    :return: The extension metadata.
    :rtype: dict
    """
    b = ApiBuilder()
    b.run()
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
# for m in get_source_code_modules(source_code_path):
    # index_template = index_template + f"   {m}\n"
# print(index_template)
