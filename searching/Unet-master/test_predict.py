from unet import *
import numpy as np
class dataProcess(object):
    def __init__(self, out_rows, out_cols, test_path = "/home/baoge/shuchuji/output", npy_path = "/home/baoge/image_inpainting/searching/Unet-master/npydata", img_type = "jpg"):
        # 数据处理类，初始化
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_test_data(self):
        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]
            img = load_img(self.test_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            # img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # img = np.array([img])
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test
if __name__ == "__main__":

    mydata = dataProcess(256,256)
    mydata.create_test_data()
    imgs_test = mydata.load_test_data()

    myunet = myUnet()
    model = myunet.get_unet()
    model.load_weights('/home/baoge/imagemask/my_unet.hdf5')
    imgs_mask_test = model.predict(imgs_test, batch_size=1,verbose=1)
    np.save('/home/baoge/image_inpainting/searching/Unet-master/results/imgs_mask_test.npy', imgs_mask_test)





    print("array to image")
    imgs = np.load('/home/baoge/image_inpainting/searching/Unet-master/results/imgs_mask_test.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("/home/baoge/image_inpainting/searching/Unet-master/results/%d.jpg" % (i))