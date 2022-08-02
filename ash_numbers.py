import os
from pprint import pprint

allFolders = os.listdir("ash-numbers/ood_eval_2022-08-01_22-00-22")

allLines = []

for folder in allFolders:
    DIR_PATH = "ash-numbers/ood_eval_2022-08-01_22-00-22/{}".format(folder)
    if not os.path.isdir(DIR_PATH): continue
    files = os.listdir(DIR_PATH)
    if "stdout" in files:
        stdout = "{}/stdout".format(DIR_PATH)
        f = open(stdout, 'r')
        lines = f.readlines()
        allLines.extend(lines[6:])
        allLines.append('\n')

with open('all-ash-numbers.txt', 'w') as f:
    for line in allLines:
        f.write("%s" % line)
