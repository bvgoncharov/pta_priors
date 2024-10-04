import numpy as np
import json

datadir = './'
with open(datadir+'15yr_wn_dict.json', 'r') as f:
    data_str = f.read()

data_dict = json.loads(data_str)

psrs = np.loadtxt('../../../params/ng15_psrs.txt',dtype=str)

for psr in psrs:
    new_file = psr + '_noise.json'
    out_dict = {}
    for key, val in data_dict.items():
        if psr in key:
            out_dict[key] = val

    with open(datadir+new_file, 'w') as fout:
        json.dump(out_dict, fout, sort_keys=True, indent=4, separators=(',', ': '))
