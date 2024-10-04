import os
import json

# Get the current directory
current_dir = os.getcwd()

# Collect all .par file names
par_files = [filename for filename in os.listdir(current_dir) if filename.endswith(".par")]

# Process file names to discard part that includes and follows "_PINT"
psrs = [filename.split('_PINT')[0] for filename in par_files]

# Load the JSON file
json_file = 'noisemodels/cp_dmgp_dips_vanilla_30nf_20240716.json'
with open(json_file, 'r') as file:
    data = json.load(file)

# Print elements of psrs not present as keys in the dictionary
missing_psrs = [psr for psr in psrs if psr not in data.keys()]

print("Missing psrs:", missing_psrs)
