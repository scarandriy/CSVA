#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import re

# • Pass a path as the first CLI argument, or default to the current directory
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

def numeric_key(path: Path):
    """
    Extract the first integer from the filename and use it for sorting.
    If no number is found, fall back to plain name-based ordering.
    """
    m = re.search(r'\d+', path.stem)
    return (int(m.group()) if m else float('inf'), path.name)

files = sorted( (p for p in folder.iterdir() if p.is_file()),
                key=numeric_key)

with open("file_list.txt", "w", encoding="utf-8") as out_file:
    for f in files:
        out_file.write(f"{f.name}\n")
