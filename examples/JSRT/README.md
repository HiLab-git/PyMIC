# Heart segmentation from 2D X-ray images

![image_example](./picture/JPCLN001.png)
![label_example](./picture/JPCLN001_seg.png)

In this example, we use 2D U-Net to segment the heart from X-Ray images. First we download the images from internet, then edit the configuration file for training and testing. During training, we use tensorboard to observe the performance of the network at different iterations. We then apply the trained model to testing images and obtain quantitative evaluation results.

If you don't want to train the model by yourself, you can download a pre-trained model [here][model_link] and jump to the `Testing and evaluation` section.

## Data and preprocessing
1. The JSRT dataset is available at the [JSRT website][jsrt_link]. It consists of 247 chest radiographs. Create a new folder as `JSRT_root`, and download the images and save them in to a single folder, like `JSRT_root/All247images`. 
2. The annotation of this dataset is provided by the [SCR database][scr_link]. Download the annotations and move the unzipped folder `scratch` to `JSRT_root/scratch`.
3. Create two new folders  `JSRT_root/image` and `JSRT_root/label` for preprocessing.
4. Set `JSRT_root` according to your computer in `image_convert.py` and run `python image_convert.py` for preprocessing. This command converts the raw image format to png and resizes all images into 256X256. The processed image and label are saved in `JSRT_root/image` and `JSRT_root/label` respectively.
5. Set `JSRT_root` according to your computer in `write_csv_files.py` and run `python write_csv_files.py` to randomly split the 247 images into training (180 images), validation (20 images) and testing (47 images) sets. The output csv files are saved in `config`.

[model_link]:https://drive.google.com/open?id=1pYwt0lRiV_QrCJe5ef9IsLf4NKyrFRRD
[jsrt_link]:http://db.jsrt.or.jp/eng.php
[scr_link]:https://www.isi.uu.nl/Research/Databases/SCR/ 

## Training
1. Edit `config/train_test.cfg` by setting the value of `root_dir` as your `JSRT_root`. Then start to train by running:
 
```bash
pymic_net_run train config/train_test.cfg
```

2. During training or after training, run `tensorboard --logdir model/unet` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where blue and red curves are for training set and validation set respectively. We can observe some over-fitting on the training set. 

![avg_dice](./picture/jsrt_avg_dice.png)
![avg_loss](./picture/jsrt_avg_loss.png)

## Testing and evaluation
1. Run the following command to obtain segmentation results of testing images. If you use [the pretrained model][model_link], you need to edit `checkpoint_name` in `config/train_test.cfg`.

```bash
mkdir result
pymic_net_run test config/train_test.cfg
```

2. Then edit `config/evaluation.cfg` by setting `ground_truth_folder_list` as your `JSRT_root/label`, and run the following command to obtain quantitative evaluation results in terms of dice. 

```bash
pymic_evaluate config/evaluation.cfg
```

The obtained dice score by default setting should be close to 94.88+/-2.72%. You can set `metric = assd` in `config/evaluation.cfg` and run the evaluation command again. You will get average symmetric surface distance (assd) evaluation results. By default setting, the assd is close to 2.08+/-1.06 pixels.

