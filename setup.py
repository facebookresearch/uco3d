from setuptools import find_packages, setup

setup(
    name="uco3d",
    version="1.0",
    packages=find_packages(exclude=["tests", "dataset_download", "examples"]),
    install_requires=[
        "sqlalchemy>=2.0",
        "pandas",
        "tqdm",
        "torchvision",
        "torch",
        "matplotlib",
        "plyfile",
    ],
)
