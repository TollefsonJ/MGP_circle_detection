# This script reads in a list of LOC files from a folder and converts that list to a dataframe
# It's designed to be run after detect_circles_LOC.py
# It's structured to read a list of files which includes input files AND the files that are output from detect_circles_LOC.py


# import the necessary packages
import glob
import pandas as pd
import numpy as np

## Construct file list
imgnames = sorted(glob.glob("input/***/*.jpg"))


# set dataframe
df = pd.DataFrame()

# Add all files to dataframe
df['file'] = pd.Series(imgnames).astype(str)

# Set "circles" variable to 0
df['circles'] = 0



# Set number of circles according to output filename from detect_circles_LOC.py
# If there isn't a number of circles, it doesn't update circle string

df['circles']= df['file'].str.extract(r'((?<=_out)\w+(?=\.jpg))')

# strip out unnecessary info from filename
df['file'] = df['file'].str.split('/').str[-1]
df['file'] = df['file'].str.split('_out').str[0]
df['file'] = df['file'].str.rstrip('_out.jpg')
df['file'] = df['file'].str.rstrip('.jpg')

# Delete the duplicate "input" listing for all "output" files
df = df.sort_values(by=['circles'])
df = df.drop_duplicates(subset='file', keep='first')

# Split filename into constitutive components
df[['set','page']] = df.file.str.split("-", expand = True)
df['city'] = df['set'].str[:5]
df['year'] = df['set'].str[-4:]
df['volume'] = df['set'].str.extract(r'((?<=_)\w+(?=_))')

# Sort by filename and save as csv
df = df.sort_values(by=['file'])
# dfout = df[['city', 'volume', 'year', 'page', 'circles']]
df.to_csv('output.csv', index=False)
