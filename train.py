#!/usr/bin/env python3
import os
import runpy
import sys


def main():
    script_path = os.path.join(os.path.dirname(__file__), "code_test.py")
    sys.argv = [script_path, *sys.argv[1:]]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
