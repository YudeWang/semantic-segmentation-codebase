# deeplabv3+voc
Here is the experiments of DeepLabv3+ on PASCAL VOC 2012 dataset.

## Dataset
Please refer to another [repository](https://github.com/YudeWang/deeplabv3plus-pytorch) for Dataset preparation, which is an ancient version of this codebase.

## Model
The ImageNet pretrained ResNet101 can be download according to code comment [here](https://github.com/YudeWang/semantic-segmentation-codebase/blob/cfed09692cc1aea2ccabe7a58eec888681eecc20/lib/net/backbone/resnet.py#L19), and please replace the pretrained model path after download.
| backbone | output stride | multi-scale & flip test | val mIoU |
|----------|---------------|-------------------------|----------|
| [xception_pytorch_imagnet](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi) | | | |
| [deeplabv3+res101](https://drive.google.com/file/d/1TAweDPn5Ohr7Ag6JvAE33fnZbVMSrzMX/view?usp=sharing) | 8 | True | 81.139% |
| [deeplabv3+xception](https://drive.google.com/file/d/1VCxu9h5SVcTLD4OnCUeWdfoUNr-MbUwW/view?usp=sharing) | 8 | True | 81.142% |

## Usage
Please modify the configration in `config.py` according to your device firstly.
```
python train.py
```
Don't forget to check test configration in `config.py` then
```
python test.py
```
