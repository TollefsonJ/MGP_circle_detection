# Use to prepare training images for analysis
# resize images to 64x64 and converts to grayscale




# First, choose set: balanced, imbalanced, all_tanks
set = "imbalanced"

from PIL import Image, ImageOps
import os
def resize_multiple_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in os.listdir(src_path):
        try:
            img=Image.open(src_path+filename)
            new_img = img.resize((64,64))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path+filename)
        except:
            continue

src_path = "training_images/" + set + "/YES_full/"
dst_path = "training_images/" + set + "/YES_resized/"
resize_multiple_images(src_path, dst_path)

src_path = "training_images/" + set + "/NO_full/"
dst_path = "training_images/" + set + "/NO_resized/"
resize_multiple_images(src_path, dst_path)