from setuptools import setup
import subprocess
import shutil
import os

# Clone and install rad
subprocess.run(["git", "clone", "--recursive", "https://github.com/keiserlab/rad.git"], check=True)
subprocess.run(["pip", "install", "./rad"], check=True)
shutil.rmtree("rad", ignore_errors=True)

setup(
    name="lsd",
    version="0.1",
    install_requires=[
        "pyarrow",
        "stringzilla",
        "chemprop==2.0.2",
    ],
)
