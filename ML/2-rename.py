import os
def rename_multiple_files(path,obj):
    i=0
    for filename in os.listdir(path):
        try:
            f,extension = os.path.splitext(path+filename)
            src=path+filename
            dst=path+obj+str(i)+extension
            os.rename(src,dst)
            i+=1
            print('Rename successful.')
        except:
            i+=1

path = "training_images/FINAL/"
obj = "YES"

rename_multiple_files(path,obj)


path = "training_images/NO2/"
obj = "NO"

rename_multiple_files(path,obj)


# move files from NO-input2 to final-input to prepare for array

import shutil
import os

source = 'training_images/NO2/'
dest1 = 'training_images/FINAL/'

files = os.listdir(source)

for f in files:
    if "jpg" in f:
        shutil.move(source+f, dest1)
