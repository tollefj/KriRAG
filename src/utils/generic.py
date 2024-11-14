# ------------------------------------------------------------------------------
# File: generic.py
# Description: environment and general utilities for KriRAG
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

import os
import subprocess
import sys
from typing import List

from dotenv import dotenv_values, load_dotenv


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def init_sqlite():
    # is_linux = sys.platform == "linux"
    # if is_linux:
    #     # first check if it's already installed
    #     if sys.modules.get("sqlite3"):
    #         return
    #     # install("pysqlite3-binary")  # invalid on docker
    #     __import__("pysqlite3")
    #     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    pass


def init_dotenv(custom_environments: List[str] = []):
    if isinstance(custom_environments, str):
        custom_environments = [custom_environments]
    if not os.path.exists(".env"):
        print("No .env file found")
        return
    load_dotenv(".env")
    for env in custom_environments:
        print(f"Overriding env with path {env}")
        load_dotenv(env, override=True)
        dotenv_keys = list(dotenv_values(dotenv_path=env).keys())
        for dk in dotenv_keys:
            print(f"Override {dk} --> {os.getenv(dk)}")