import os
import re

def update_file(fname, path_to_model):
    print("changing "+ fname)
    with open(fname, 'r') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines):
        if '.include' in line:
            regex = re.compile("\.include\s*\"(.*?45nm\_bulk\.txt)\"")
            found = regex.search(line)
            if found:
                lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)

    with open(fname, 'w') as f:
        f.writelines(lines)
        f.close()

if __name__ == '__main__':
    cur_fpath = os.path.realpath(__file__)
    parent_path = os.path.abspath(os.path.join(cur_fpath, os.pardir))
    netlist_path = os.path.join(parent_path, 'netlist')
    spice_model = os.path.join(parent_path, 'spice_models/45nm_bulk.txt')

    for root, dirs, files in os.walk(netlist_path):
        for f in files:
            if f.endswith(".cir"):
                update_file(fname=os.path.join(root, f), path_to_model=spice_model)