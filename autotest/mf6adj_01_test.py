import os
import sys
import platform
import shutil
import string
import time
import numpy as np
import pandas as pd

sys.path.insert(0,".")
import flopy

if "linux" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "linux", "libmf6.so")
    mf6_bin = os.path.join("..", "bin", "linux", "mf6")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
else:
    lib_name = os.path.join("..", "bin", "win", "libmf6.dll")
    mf6_bin = os.path.join("..", "bin", "win", "mf6.exe")

sys.path.insert(0,os.path.join(".."))


def basic_freyberg():
    org_d = os.path.join("models","freyberg")
    new_d = "freyberg"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)
    shutil.copy2(lib_name,os.path.join(new_d,os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin,os.path.join(new_d,os.path.split(mf6_bin)[1]))

    os.chdir(new_d)
    os.system("mf6")
    os.chdir("..")

    import mf6adj



if __name__ == "__main__":
    basic_freyberg()

