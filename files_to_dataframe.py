# This script reads in a list of LOC files from a folder and converts that list to a dataframe
# It's designed to be run after detect_circles_LOC.py
# It's structured to read a list of files which includes input files AND the files that are output from detect_circles_LOC.py


# import the necessary packages
import glob
import pandas as pd
import numpy as np

## Construct file list
imgnames = sorted(glob.glob("input/*.jpg"))


# set dataframe
df = pd.DataFrame()

# Add all files to dataframe
df['file'] = pd.Series(imgnames).astype(str)

# Set "circles" variable to 0
df['circles'] = 0

# Set circles=1 if the filename is identified as an output from detect_circles_LOC.py
df.loc[df['file'].str.contains("_out"), 'circ'] = 1

# strip out unnecessary info from filename
df['file'] = df['file'].str.split('/').str[-1]
df['file'] = df['file'].str.rstrip('_out.jpg')
df['file'] = df['file'].str.rstrip('.jpg')

# Delete the duplicate "input" listing for all "output" files
df = df.sort_values(by=['circ'])
df = df.drop_duplicates(subset='file', keep='last')
df[['set','number']] = df.file.str.split("-", expand = True)

# Sort by filename and save as csv
df = df.sort_values(by=['file'])
df.to_csv('output.csv', index=False)


