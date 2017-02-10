#!/usr/bin/env python

"""
This script will extract standalone runnable Python script from Python notebook (*.ipynb) files. To execute script, place it inside the directory of python machine learning book, and pass the chapter number as the only argument.
"""

import sys
import json
import re
import os


def main():
    nb_dir = r'/home/lawrenceli/Workplace/python-machine-learning-book/code'
    chapters = os.listdir(nb_dir)
    chapters = filter(lambda s: re.match(r'ch\d+', s), chapters)
    for ch in chapters:
        obj = json.loads(open(os.path.join(nb_dir, ch, '{}.ipynb'.format(ch))).read())
        if not os.path.isdir(ch):
            os.mkdir(ch)
        fid = open(os.path.join(ch, '{}.all.py'.format(ch)), 'w')
        for i, cell in enumerate(obj['cells']):
            for ln in cell['source']:
                if re.match(r'^\s+$', ln):
                    ln = '\n'
                fid.write(ln.encode('utf-8'))
            fid.write('\n')
        fid.close()


if __name__ == '__main__':
    main()
