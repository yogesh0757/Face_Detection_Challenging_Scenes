# Face_Detection_Challenging_Scenes in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [MFRNet: Face_Detection_Challenging_Scenes](https://arxiv.org/abs/1905.00641). Model size only 50M, We achieve better result with low computation FLOPS.

## WiderFace Val Performance in single scale When using MFRNet as backbone net.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (1200 to upto 1600) | 94.85 % | 93.50% | 89.30% |
| Pytorch (original image scale) | 95.48% | 94.04% | 85.43% |

## FDDB Performance.
| FDDB(pytorch) | performance |
|:-|:-:|
| MRFNet | 98.64% |

### Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [TensorRT](#tensorrt)
- [References](#references)

## Installation
##### Clone and install
1. git clone https://github.com/yogesh0757/Face_Detection_Challenging_Scenes.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```

I used !python test_widerface.py --network resnet50 --cpu to generate the txt file.
Then I downloaded the result in file.zip which contains the test result of validaion dataset in txt format.
After this step I downloaded file.zip and moved it to widerface_evaluate/widerface_txt.
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
In order to plot the precision vs. recall curve I needed to make some modification in evaluation.py.Appended from line 274-277 as:
plt.plot(recall, propose, color ='tab:blue')
        plt.savefig("save_curve.png",bbox_inches="tight")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
Before that I imported the python library matplotlib as plt.
The precision vs. recall curve was saved in save_curve.png .


3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)


<p align="center"><img src="curve/1.jpg" width="640"\></p>

## TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
REFERENCES:
1.https://drive.google.com/file/d/1iUYvk33zxV2dU-sG6EWUd4KW9YPzhc2n/view?usp=sharing -This is the google drive link of WIDER_VAL.zip which contains the validation images and can be accessed directly from my drive using this link.
2.https://drive.google.com/file/d/1rX11lpo3xyN8JsJ56r10KPu9IFALRQRX/view?usp=sharing -This is the google drive link of label.txt file of the validation dataset.It will be downloaded and saved as wider_val.txt file in the google colab notebook.
3.https://drive.google.com/file/d/1pLcsCaDSsfTrG01pm8WR5dCu0dJvzbKR/view?usp=sharing -This is the google drive link of file.zip which contains the result in txt format after testing the model with validation dataset.
4.https://colab.research.google.com/drive/1lReOo_0j50waqspBcE8zodT4jr3i9CJF?usp=sharing -This is the google drive link of my Retinaface_Pytorch.ipynb google colab notebook.My notebook can be directly accessed using this link.
