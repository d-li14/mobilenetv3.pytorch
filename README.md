# PyTorch Implemention of MobileNet V3
Reproduction of MobileNet V3 architecture as described in [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) by Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.

# Requirements
## Dataset
Download the ImageNet dataset and move validation images to labeled subfolders.
To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Models
| Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| MobileNetV3-Large | 5.145M       | 245.58 |                            |
| [MobileNetV2 1.0](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2-0c6065bc.pth)  | 3.504M       | 300.79 | 72.192 / 90.534            |
| MobileNetV3-Small | 3.112M       |  57.08 |                            |
| [MobileNetV2 0.35](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.35-b2e15951.pth)  | 1.677M       |  59.29 | 60.092 / 82.172        |

*Note: The implemented architecture follows Table 1 and 2 in the paper, yet architectural details are vaguely described, rendering mismatches of both parameters and complexity.*

```python
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

net_large = mobilenetv3_large()
net_small = mobilenetv3_small()

# pretrained models will come soon
net_large.load_state_dict(torch.load('pretrained/mobilenetv3-large.pth'))
net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small.pth'))
```

# Citation
```
@ARTICLE{2019arXiv190502244H,
       author = {{Howard}, Andrew and {Sandler}, Mark and {Chu}, Grace and
         {Chen}, Liang-Chieh and {Chen}, Bo and {Tan}, Mingxing and
         {Wang}, Weijun and {Zhu}, Yukun and {Pang}, Ruoming and
         {Vasudevan}, Vijay and {Le}, Quoc V. and {Adam}, Hartwig},
        title = "{Searching for MobileNetV3}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = "2019",
        month = "May",
          eid = {arXiv:1905.02244},
        pages = {arXiv:1905.02244},
archivePrefix = {arXiv},
       eprint = {1905.02244},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190502244H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
