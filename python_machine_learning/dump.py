#!/usr/bin/env python

"""
This script will extract standalone runnable Python script from Python notebook (*.ipynb) files. To execute script, place it inside the directory of python machine learning book, and pass the chapter number as the only argument.
"""

import sys
import json
import re


def main():
    dst = sys.argv[1]
    obj = json.loads(open('{}/{}.ipynb'.format(dst, dst)).read())
    for i, cell in enumerate(obj['cells']):
        fid = open('{}/cell_{}.py'.format(dst, i), 'w')
        for ln in cell['source']:
            if re.match(r'^\s+$', ln):
                ln = '\n'
            fid.write(ln.encode('utf-8'))


if __name__ == '__main__':
    main()
