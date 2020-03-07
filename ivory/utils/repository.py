import importlib
import os
import subprocess


def repo_url(name: str) -> str:
    if name.startswith("scratch"):
        url = "https://github.com/oreilly-japan/deep-learning-from-scratch"
        if name.endswith("2"):
            url += "-2"
    elif name.startswith("nnabla"):
        url = f"https://github.com/sony/{name}"
    elif name == "keras":
        url = "https://github.com/fchollet/deep-learning-with-python-notebooks"
    else:
        raise ValueError("Unknown name")
    return url


def repo_directory(name: str) -> str:
    url = repo_url(name)
    reponame = url.split("/")[-1]
    directory = os.path.normpath(os.path.expanduser("~/.ivory"))
    return os.path.join(directory, reponame)


def clone(name: str) -> None:
    url = repo_url(name)
    directory, reponame = os.path.split(repo_directory(name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    curdir = os.path.abspath(os.path.curdir)
    os.chdir(directory)
    try:
        if not os.path.exists(reponame):
            subprocess.run(["git", "clone", url])
    finally:
        os.chdir(curdir)


def import_module(name: str):
    repo, *paths, module_str = name.split("/")
    directory = repo_directory(repo)
    directory = os.path.join(directory, *paths)

    curdir = os.path.abspath(os.path.curdir)
    os.chdir(directory)
    try:
        module = importlib.import_module(module_str)
    finally:
        os.chdir(curdir)

    if not hasattr(module, "name"):
        module.name = name  # type: ignore
    return module


def run(name: str):
    repo, *paths, script = name.split("/")
    directory = repo_directory(repo)
    directory = os.path.join(directory, *paths)

    curdir = os.path.abspath(os.path.curdir)
    os.chdir(directory)
    try:
        with open(script, encoding="utf-8") as f:
            source = f.read()
        exec(source)
    finally:
        os.chdir(curdir)


def getsource(name: str) -> str:
    repo, *paths = name.split("/")
    directory = repo_directory(repo)
    path = os.path.join(directory, *paths)
    with open(path, encoding="utf-8") as f:
        return f.read()
