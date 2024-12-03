from setuptools import setup, find_packages

setup(
    name="uco3d",
    version="1.0",
    packages=find_packages(exclude=["tests", "dataset_download"]),
    install_requires=[],
)
