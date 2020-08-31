
# resize images to 64x64 and converts to grayscale


from PIL import Image, ImageOps
import os
def resize_multiple_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in os.listdir(src_path):
        try:
            img=Image.open(src_path+filename)
            gray = ImageOps.grayscale(img)
            new_img = gray.resize((1433,2066))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path+filename)
            print('Resized, grayed, and saved {} successfully.'.format(filename))
        except:
            continue



src_path = "analysis_images/input/"
dst_path = "analysis_images/resized/"
resize_multiple_images(src_path, dst_path)
