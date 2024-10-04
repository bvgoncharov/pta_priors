import os

# Get the current directory
current_dir = os.getcwd()

# Iterate over all files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".par"):
        # Open the file for reading and writing
        with open(filename, 'r+') as file:
            lines = file.readlines()
            # Go back to the beginning of the file
            file.seek(0)
            for line in lines:
                if line.startswith("DMX"):
                    file.write("# " + line)
                else:
                    file.write(line)
            # Truncate the file to the new size
            file.truncate()
