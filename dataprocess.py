import io
import pandas as pd
import os
from glob import glob
import re
import string


file_loc = "Labels.xlsx"
df = pd.read_excel(file_loc, sheet_name='Sheet1')
sub_name = df["course"].tolist()
path_name = df["Path"].tolist()

base = "C:\\Users\\vnitin\\Documents\\dataset_extractor\\"


"""
# Folder renaming code

for old_name in os.listdir(base):
    #print(old_name)
    #for path, course in zip(path_name,sub_name):
    #    if(old_name == path):
    old_path = os.path.join(base, old_name)
    #new_path = os.path.join(base, course)
    replaced = os.rename(old_path, old_path.replace(' ','_') )
    print(old_path,replaced)
"""


# Convert all file into text files

read_pdf =  []
all_pdf_path = []
for path in os.listdir(base):
    #print(path)
    for file in os.listdir(os.path.join(base, path)):
        #print(file)
        all_pdf_path = os.path.join((os.path.join(base, path)),file)
        text_path = all_pdf_path.replace('pdf', 'txt')
        #print(text_path)
        os.system("py ./pdf2txt.py -o {0} {1}".format(text_path,all_pdf_path))
        #print("Success",text_path)
    print("Done",path)



