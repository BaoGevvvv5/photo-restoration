import numpy as np
import glob
import os
data_path = "/home/baoge/imagemask/mask"
img_type = "jpg"
imgs = glob.glob(data_path+"/*."+img_type)
#print(imgs)
for imgname in imgs:
    midname = imgname[imgname.rindex("/") + 1:]
    lastname = midname[midname.rindex("_") + 1:]
    lastname = lastname[0:8]
    change = int(lastname)
    change = str(change)
    os.rename(imgname,change+".jpg")
#str = "/home/baoge/image_inpainting/searching"
#name = str.rindex("/")
#name = str[str.rindex("/")+1:]
#print(name)