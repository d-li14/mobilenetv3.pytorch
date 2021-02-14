# PyTorch Implementation of MobileNet V3
Reproduction of MobileNet V3 architecture as described in [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) by Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.

# Requirements
## Dataset
Download the ImageNet dataset and move validation images to labeled subfolders.
To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Training recipe
* *batch size* 1024
* *epoch* 150
* *learning rate* 0.4 (ramps up from 0.1 to 0.4 in the first 5 epochs)
* *LR decay strategy* cosine
* *weight decay* 0.00004
* *dropout rate* 0.2 (0.1 for Small-version 0.75)
* *no weight decay* biases and BN
* *label smoothing* 0.1 (only for Large-version)

# Models
| Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| [MobileNetV3-Large 1.0](https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-1cd25616.pth) | 5.483M | 216.60 | 74.280 / 91.928 |
| [MobileNetV3-Large 0.75](https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-0.75-9632d2a8.pth) | 3.994M | 154.57 | 72.842 / 90.846 |
| [MobileNetV3-Small 1.0](https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-55df8e1f.pth) | 2.543M |  56.52 | 67.214 / 87.304 |
| [MobileNetV3-Small 0.75](https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-0.75-86c972c3.pth) | 2.042M |  43.40 | 64.876 / 85.498 |


```python
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

net_large = mobilenetv3_large()
net_small = mobilenetv3_small()

net_large.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth'))
```

# Citation
```
@InProceedings{Howard_2019_ICCV,
author = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
title = {Searching for MobileNetV3},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
If you find this implementation helpful in your research, please also consider citing:
```
@InProceedings{Li_2019_ICCV,
author = {Li, Duo and Zhou, Aojun and Yao, Anbang},
title = {HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
