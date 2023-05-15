#!/usr/bin/env python3

import json
_dict = {
    "Cosmology": 
    {
        "h0":70.0,
        "om0":0.3,
        "l0":0.7
    },
    "Templates":
    {
        "name1":"/path/to/file1.sed",
    },

    "Filters":
    {
        "name1":"/path/to/file1.filt",
    },

    "Extinctions":
    {
        "name1":"/path/to/file1.ext",
    },

    "Estimator":"chi2",
    "Dataset":
    {
        "path":"/path/to/data.set",
        "type":"F"
    }
}

with open("defaults.json", "w") as wf:
    json.dump(_dict, wf)


