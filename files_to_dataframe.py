# this script reads in a list of

#### FOLDER STRUCTURE###
# level 1: input folder, this script
# input folder: a collection of folders with images (images must be in folders starting with "saveTo"; no sub-folders)
# place output folder within input folder


# import the necessary packages

import glob
import pandas as pd
import numpy as np

## Get all the png image in the PATH_TO_IMAGES. Input whatever folder you're actually using.
## including /saveTo**/ also searches one level of sub-directories below "input" that start with saveTo
imgnames = sorted(glob.glob("input/***/*.jpg"))


# add outputs and non-outputs to a dataframe of all results
df = pd.DataFrame()

# positive outputs
df['file'] = pd.Series(imgnames).astype(str)
df['circ'] = 0
df.loc[df['file'].str.contains("_out"), 'circ'] = 1

# strip out unnecessary info
df['file'] = df['file'].str.split('/').str[-1]
df['file'] = df['file'].str.rstrip('_out.jpg')
df['file'] = df['file'].str.rstrip('.jpg')

# delete the negative inputs to keep only positive outputs for those file
df = df.sort_values(by=['circ'])
df = df.drop_duplicates(subset='file', keep='last')
df[['set','number']] = df.file.str.split("-", expand = True)

# sort and output as csv
df = df.sort_values(by=['file'])
df.to_csv('output.csv', index=False)

# now, look at dataframe to see if there are any folders with a lot of positives.
# do so by getting the mean of "circ", grouped by "set"
# This incidates they have a compass visible, and you need to run the compass script instead.
