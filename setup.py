import pathlib
import setuptools


def get_version():
    version = "0.0.0"
    with pathlib.Path("VERSION.txt").open() as ver_file:
        version = ver_file.read()
    return version


setuptools.setup(
    version=get_version()
)
