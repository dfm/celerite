#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import re
import shutil
from datetime import date
from subprocess import check_call

datestamp = date.today().strftime("%m%d")
TMPDIR = "fore" + datestamp
outfn = TMPDIR+".tar"
tex = open("ms.tex", "r").read()

try:
    os.makedirs(TMPDIR)
except os.error:
    pass

def rename(fn):
    a, b = os.path.split(fn)
    strt, end = os.path.split(a)
    if len(strt):
        b = end + "-" + b
    return b, fn

def find_figures(tex):
    return re.findall("includegraphics(?:.*?){(.*)}", tex)

def find_includes(tex):
    return re.findall("include{(.*)}", tex)

def find_inputs(tex):
    return re.findall("input{(.*)}", tex)

for fn in find_includes(tex):
    txt = open(fn + ".tex").read()
    tex = tex.replace("\\include{{{0}}}".format(fn), txt)

for fn in find_inputs(tex):
    txt = open(fn + ".tex").read()
    tex = tex.replace("\\input{{{0}}}".format(fn), txt)

for name, loc in map(rename, find_figures(tex)):
    shutil.copyfile(loc, os.path.join(TMPDIR, name))
    tex = tex.replace(loc, name)

shutil.copyfile("aastex6.cls", os.path.join(TMPDIR, "aastex6.cls"))
shutil.copyfile("aasjournal.bst", os.path.join(TMPDIR, "aasjournal.bst"))
shutil.copyfile("ms.bbl", os.path.join(TMPDIR, "ms.bbl"))
open(os.path.join(TMPDIR, "ms.tex"), "w").write(tex)
check_call(" ".join(["cd", TMPDIR+";",
                     "tar", "-cf", os.path.join("..", outfn), "*"]),
           shell=True)
shutil.rmtree(TMPDIR)

print("Wrote file: '{0}'".format(outfn))
