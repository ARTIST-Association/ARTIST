# -*- coding: utf-8 -*-
"""
    Setup file for artist.
    Use setup.cfg to configure your project.
"""
import sys
import os
import subprocess

from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

if __name__ == "__main__":
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "torch"])
    os.system("export PYTORCH3D_NO_EXTENSION=1")
    setup()
