import os
import shutil


def cache_dir(*path, rmtree=False):
    basedir = os.path.expanduser("~/.ivory/datasets")
    path = os.path.normpath(os.path.join(basedir, *path))
    if not os.path.exists(path):
        os.makedirs(path)
    elif rmtree:
        shutil.rmtree(path)
        os.mkdir(path)
    return path


def cache_file(*path, rmtree=False):
    directory, path = os.path.split(os.path.join(*path))
    directory = cache_dir(directory, rmtree=rmtree)
    return os.path.normpath(os.path.join(directory, path))
