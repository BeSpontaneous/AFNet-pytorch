# Look More but Care Less in Video Recognition (NeurIPS 2022)

<div align="left">
    <a><img src="fig/smile.png"  height="70px" ></a>
    <a><img src="fig/neu.png"  height="70px" ></a>
</div>

<<<<<<< Updated upstream
arXiv | Primary contact: [Yitian Zhang](mailto:markcheung9248@gmail.com)
=======
[arXiv](https://arxiv.org/abs/2211.09992) | Primary contact: [Yitian Zhang](mailto:markcheung9248@gmail.com)
>>>>>>> Stashed changes

<div align="center">
  <img src="fig/intro.jpeg" width="850px" height="150px">
</div>

Comparisons between existing methods and our proposed Ample and Focal Network (AFNet). Most existing works reduce the redundancy in data at the beginning of the deep networks which leads to the loss of information. We propose a two-branch design which processes frames with different computational resources within the network and preserves all input information as well.

<<<<<<< Updated upstream
## Code will be available soon. (seriously:monkey::monkey::monkey:)
=======

## Requirements
- python 3.7
- pytorch 1.7.0
- torchvision 0.9.0


## Datasets
Please follow the instruction of [TSM](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) to prepare the Something-Something V1/V2 dataset.


## Pretrained Models
Here we provide the pretrained AF-MobileNetv3, AF-ResNet50, AF-ResNet101 on ImageNet and all the pretrained models on Something-Something V1 dataset.

### Results on ImageNet
Checkpoints are available through the [link](https://drive.google.com/drive/folders/1UzSckmKnwmgwWObF2_YxpkAIZ2k2mcHL?usp=share_link).
| Model           | Top-1 Acc.    |  GFLOPs  |
| --------------- | ------------- | ------------- |
| AF-MobileNetv3  | 72.09% |  0.2  |  
| AF-ResNet50     | 77.24% |  2.9  |
| AF-ResNet101    | 78.36% |  5.0  |

### Results on Something-Something V1
Checkpoints and logs are available through the [link](https://drive.google.com/drive/folders/1-xmE6T6OADmDkkzJr4iM1vCJbA4ofcSO?usp=share_link).

**Less is More**:
| Model | Frame | Top-1 Acc. |  GFLOPs  |
| --------------- | --------------- | ------------- | ------------- |
| TSN  | 8 | 18.6% | 32.7 |
| AFNet(RT=0.50) | 8 | 26.8% |  19.5  |
| AFNet(RT=0.25) | 8 | 27.7% |  18.3  |


**More is Less**:  
| Model | Backbone | Frame | Top-1 Acc. |  GFLOPs  |
| --------------- | --------------- | ------------- |------------- | ------------- |
| TSM  | ResNet50 | 8 | 45.6% | 32.7 |
| AFNet-TSM(RT=0.4)  | AF-ResNet50  | 12 | 49.0% | 27.9 |
| AFNet-TSM(RT=0.8)  | AF-ResNet50  |  12 |49.9% | 31.7 |
| AFNet-TSM(RT=0.4)  | AF-MobileNetv3  | 12 | 45.3% | 2.2 |
| AFNet-TSM(RT=0.8)  | AF-MobileNetv3  | 12 | 45.9% | 2.3 |
| AFNet-TSM(RT=0.4)  | AF-ResNet101  | 12 | 49.8% | 42.1 |
| AFNet-TSM(RT=0.4)  | AF-ResNet101  | 12 | 50.1% | 48.9 |


## Training AFNet on Something-Something V1
1. Specify the directory of datasets with `root_dataset` in `train_sth.sh`. 
2. Please download pretrained backbone on ImageNet from [Google Drive](https://drive.google.com/drive/folders/1UzSckmKnwmgwWObF2_YxpkAIZ2k2mcHL?usp=share_link).
3. Specify the directory of the downloaded backbone with `path_backbone` in `train_sth.sh`.
4. Specify the ratio of selected frames with `rt` and run `bash train_sth.sh`.



## Evaluate pretrained models on Something-Something V1
**Note that there is a small variance during evaluation because of Gumbel-Softmax and the testing results may not align with the numbers in our paper. We provide the logs in Tab 2 for verification.**
1. Specify the directory of datasets with `root_dataset` in `eval_sth.sh`. 
2. Please download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1-xmE6T6OADmDkkzJr4iM1vCJbA4ofcSO?usp=share_link).
3. Specify the directory of the pretrained model with `resume` in `eval_sth.sh`.
4. Run `bash eval_sth.sh`.



## Reference
If you find our code or paper useful for your research, please cite:
```
@article{zhang2022look,
  title={Look More but Care Less in Video Recognition},
  author={Zhang, Yitian and Bai, Yue and Wang, Huan and Xu, Yi and Fu, Yun},
  journal={arXiv preprint arXiv:2211.09992},
  year={2022}
}
```
>>>>>>> Stashed changes
