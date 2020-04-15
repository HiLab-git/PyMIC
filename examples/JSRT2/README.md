# Heart segmentation from 2D X-ray images using customized CNN

![image_example](../JSRT/picture/JPCLN001.png)
![label_example](../JSRT/picture/JPCLN001_seg.png)

In this example, we show how to use a customized CNN to segment the heart from X-Ray images. The configurations are the same as those in the `JSRT` example except the network structure. 

The customized CNN is detailed in `my_net2d.py`, which is a modification of the 2D UNet. In this new network, we use a residual connection in each block. 

To use the customized CNN, we also write a customized main function in `jsrt_train_infer.py`, where we import a TrainInferAgent from PyMIC and set the network as our customized CNN.

## Data and preprocessing
1. Data preprocessing is the same as that in the the `JSRT` example. Please follow that example for details.

## Training
1. Edit `config/train_test.cfg` by setting the value of `root_dir` as your `JSRT_root`, and start to train by running:
 
```bash
export PYTHONPATH=$PYTHONPATH:your_path_of_PyMIC
python jsrt_train_infer.py train config/train_test.cfg
```

2. During training or after training, run `tensorboard --logdir model/my_net2d` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where blue and red curves are for training set and validation set respectively. 

![avg_dice](./picture/jsrt2_avg_dice.png)
![avg_loss](./picture/jsrt2_avg_loss.png)

## Testing and evaluation
1. Edit the `testing` section in `config/train_test.cfg`, and run the following command for testing:
 
```bash
python jsrt_train_infer.py test config/train_test.cfg
```

2. Edit `config/evaluation.cfg` by setting `ground_truth_folder_list` as your `JSRT_root/label`, and run the following command to obtain quantitative evaluation results in terms of dice.

```
python ../../pymic/util/evaluation.py config/evaluation.cfg
```

The obtained dice score by default setting should be close to 94.61+/-2.84%. 
