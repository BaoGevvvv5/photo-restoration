# old-photo-restoration

# 旧照片修复项目

## 旧照片修复项目主要由两个部分组成

### 1. 图片破损修复

> 该部分由两个深度学习网络组成，一个网络用来识别照片上的缺陷，一个用来修复照片的缺陷。我之前已经完成了修复部分的神经网络，现在重心放在识别照片缺陷的的网络上。
>
> > 查阅相关文献，决定采用Unet网络进行切割，但是照片训练集比较难找，准备通过随机mask生成来产生照片缺陷。

### 2. 图片超分辨率

> 这一块不太熟悉，但是查阅了相关论文，看起来可以实施，由于时间限制，暂时先只做512*512像素的图片超分辨率，后期看看能不能对视频进行处理。



## 下面是inpaint图片修复部分的readme文档，我把文件夹中的旧的合并到新readme文件中

# 代码环境

- 安装python3
- 安装tensorflow 1.x（注意不要最新2.0版本）
- 安装tensorflow工具包 Neurogym

# 训练前的准备


1. 数据分包

   - 代码里包含我写好的脚本 gen_flist.py 

     运行

     ```bash
     python gen_flist.py --folder_path folderpath --train_filename train_filename --validation_filename validation_filename
     ```

   - 这个脚本能够自动切分数据集成训练集和验证集。使用前请修改default地址。

     ``` python
     parser.add_argument('--folder_path',default='default',type=str,help='The folder path')
     parser.add_argument("--train_filename",default='default',type=str,help='The train filename')
     parser.add_argument('--validation_filename', default='default', type=str,help='The validation filename.')
     ```

2. mask图片以及masked图片生成

   - 参考我代码里的 imagecreate.py 脚本。

     以生成矩形mask块为例，运行

     ``` bash
     python imagecreate.py --input_dirimg input_dirimg --output_dirmask output_dirmask --output_dirmasked output_dirmasked --HEIGHT X --WIDTH Y
     ```

     该代码将 input_dirimg 地址下图片随机生成矩形mask块，高度为X宽度为Y，并将被遮挡的masked图片保存在 output_dirmasked 地址下，将生成的mask图片保存在 output_dirmask 地址下。

     代码效果如下：

     运行

     ```bash
     imagecreate.py --input_dirimg /home/baoge/imagemask/input --output_dirmask /home/baoge/imagemask/mask --output_dirmasked /home/baoge/imagemask/imgmasked --HEIGHT 64 --WIDTH 64
     ```

     输出文件夹：

     生成的随机masked图片。

     生成随机的mask，对应相应的位置。

   - imagecreate.py 脚本包含其他很多我已经写好的mask生成函数。

     包括批量生成mask图片，单张生成，随即不规则掩膜（随机线，随机椭圆形，随机圆形等），使用时请修改最下方main函数或者调用特定函数。

     ```python
     if __name__ == '__main__':
         config = parser.parse_args()
         get_path(config)
         # 单张图像生成mask
         # img = './data/test.jpg'
         # masked_img,mask = load_mask(img,config,True)
         img2maskedImg(config.input_dirimg)
     
     ```




# 训练

- 修改 inpaint.yml ，修改里面的默认数据，包括dataflist存放地址等。

- 训练模型

运行 

```bash
python train.py
```

如果需要继续训练，请修改inpaint.yml里的MODEL_RESTORE存放地址。

并运行

```bash
python train.py
```

- 测试

运行

```bash
python test.py --image maskedimage --mask maskimage --output outputimage --checkpoint model_logs/your_model_dir
```

输出模型（我把自己的模型放在了 logs/3/ 文件夹下）：

# 训练后的数据收集

- 这里我主要准备了两个脚本可以运行

1. metrics.py 

+ 全局比较图片的修复效果，将修复后的图片与原图进行比较。

运行

```bash
python metrics.py --data-path inputimage --output-path outputimage
```

运行效果如下：

2. metrics_part.py

+ 只比较图片的修复区域的效果。只取修复区域的图像进行比较。

运行

```bash
python metrics_part.py --data-path inputimage --output-path outputimage --mask-path maskimage
```

# 模型结果

- 我将之前留存的具体数据保存在datavalue.py 文件中作为参考。


# 参考文献

1.  Generative Image Inpainting with Contextual Attention https://arxiv.org/abs/1801.07892

