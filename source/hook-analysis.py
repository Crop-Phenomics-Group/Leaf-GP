# this file is used to hook hidden imports when using PyInstaller when packaging the program as an executable.
# PyInstaller will import this file if analysis.py is imported
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("skimage.io._plugins")
hiddenimports = collect_submodules('skimage.io._plugins')
print("----------- loaded hook file based on analysis.py -----------")