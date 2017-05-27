#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import json

stats_rows = []
for i, example in enumerate(["simulated/correct", "simulated/wrong-qpo",
                             "rotation/rotation", "astero/astero",
                             "transit/transit"]):
    with open(example + ".json") as f:
        data = json.load(f)

    data["ntot"] = data["nwalkers"] * (data["nburn"] + data["nsteps"])
    data["direct_time"] *= 1000
    data["time"] *= 1000
    stats_rows.append(
        (r"{0} & {N} & {J} & {direct_time:.2f} & {time:.2f} & {ndim} & {ntot} "
         r"& {neff:.0f} \\")
        .format(i+1, **data)
    )

    with open(example + "-params.json") as f:
        params = json.load(f)

    rows = []
    for p in params:
        if len(p[1]) == 1:
            rows.append(r"{0} & {1} \\"
                        .format(p[0], p[1][0]))
        elif len(p[1]) == 2:
            rows.append(r"{0} & $\mathcal{{U}}({1[0]},\,{1[1]})$ \\"
                        .format(*p))
        else:
            assert False

    with open(example + "-params.tex", "w") as f:
        f.write("\\begin{floattable}\n")
        f.write("\\begin{deluxetable}{cc}\n")
        f.write("\\tablecaption{{The parameters for Example {0}. "
                "\label{{tab:example-{0}-params}}}}\n".format(i+1))
        f.write("\\tablehead{\colhead{parameter} & \colhead{prior}}\n")
        f.write("\\startdata\n")
        f.write("\n".join(rows))
        f.write("\n\\enddata\n")
        f.write("\end{deluxetable}\n")
        f.write("\end{floattable}\n")


with open("example-stats.tex", "w") as f:
    f.write("\\begin{floattable}\n")
    f.write("\\begin{deluxetable}{cccccccc}\n")
    f.write("\caption{The scaling and convergence stats for each example. "
            "\label{tab:example-stats}}\n")
    f.write("\\tablehead{\colhead{example} & \colhead{$N$} & \colhead{$J$} & "
            "\colhead{direct cost} & \colhead{\celerite\ cost} & "
            "\colhead{$D$} & "
            "\colhead{steps} & \colhead{effective samples}\\\\\n")
    f.write("&&& \colhead{ms} & \colhead{ms} &&&}\n")
    f.write("\\startdata\n")
    f.write("\n".join(stats_rows))
    f.write("\n\\enddata\n")
    f.write("\end{deluxetable}\n")
    f.write("\end{floattable}\n")
