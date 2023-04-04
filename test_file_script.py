import re

filename = "test3.txt"
with open(filename, "r") as f:
    lines = f.readlines()
    
pattern = re.compile(":.*$")
new_lines = [re.sub(pattern, "", line) for line in lines]

with open(filename, "w") as f:
    f.writelines(new_lines)