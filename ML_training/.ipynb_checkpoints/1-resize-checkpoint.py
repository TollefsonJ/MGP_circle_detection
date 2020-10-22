# resize images to 64x64 and converts to grayscale


from PIL import Image, ImageOps
import os
def resize_multiple_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in os.listdir(src_path):
        try:
            img=Image.open(src_path+filename)
            gray = ImageOps.grayscale(img)
            new_img = gray.resize((80,80))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path+filename)
            # print('Resized, grayed, and saved {} successfully.'.format(filename))
        except:
            continue



src_path = "training_images/YES/"
dst_path = "training_images/FINAL/"
resize_multiple_images(src_path, dst_path)

src_path = "training_images/NO1/"
dst_path = "training_images/NO2/"
resize_multiple_images(src_path, dst_path)
