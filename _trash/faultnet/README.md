#### Fault picking with 3D-Unets

#### Meta-overview
PyTorch implementation of various 3d-unets for fault picking. We use [Xinming Wu's](https://github.com/xinwucwp/faultSeg) repository to get the training data (200 images and labels of size 128x128x128). We set aside 20 images as a holdout test set for accuracy benchmarking. These are the 20 images in the validation set of Wu's original implementation, but we keep these now as unseen test-set. The models with  IOU reported are:
1. VNet original
2. VNet custom modification
3. UNet-Res1
4. UNet-Res2  
5. UNet-DS

The current best model, with code released, in this repo is (UNet-Res1) which siginificantly outperforms the original Wu model and both VNets in terms of iou on the test set. The model UNet-Res2 is yet to be released. Also demonstrated is some tricks in training, in this case a simple knowledge distillation scheme. 

TO BE ADDED : Recurrent-UNets, Attention UNets 

#### Knowledge Distillation (KD)

* Knowledge distillation is supported via trainKD.py. Noticeable uplift is observed with KD (see iou list) at the cost of an additional training run (Note the distilled network does not need the full epoch-run as the teacher). Both teacher and student are identical for the KD here. Also no temperature is used. Typically temperature should be used, but you need to play around with the value (10-20). We use the simplest implementation for KD. Better guidance can be provided by using mean-teacher, pi-model, curriculum-training etc. Still the simplest implementation improves Mean-IOU of UNet-Res1 from 0.7003 to 0.7434. The KL-Div term is turned off during training as it needs more testing. Note the KD-pass can be iterated over iseveral times by constantly changing the teacher. Pseudo-labeling is used as well. The main idea of KD is [here](https://arxiv.org/abs/1503.02531), though they are numerous ways of going about it (data dependent).

* Note 1: In the KD-pass the teacher is in eval-mode, some recent work suggests having teacher in train mode as well to use INorm and Dropout params similar to training. This has not been tested here. The mse-loss can also be computed differently. 

* Note 2 : TYpically the simple KD here will not handle test data with severe distribution shifts (eg. train on RTM images and predict on KDM images). 

* Note 3: Ensembling several KD checkpoints and the original teacher will improve accuracy further.  

* TODO: Teacher-student consistency. 

#### Current model IOUs

| Filename | WU | VNet(CE+Dice) | VNet-Modified | UNet-DS (TTA *SOTA*) | UNet-Res1(CE) | Unet-Res2 | Unet-Res2(TTA) **SOTA** | UNet-Res1-KD |  
| -- | -- | -- | -- | -- | -- | -- | -- | -- |  
| 0 | 0.695 | 0.672 | 0.711 | 0.800 | 0.764 | 0.801 | 0.814 | 0.812 | 
| 1 | 0.701 | 0.721 | 0.761 | 0.824 | 0.751 | 0.809 | 0.824 | 0.796 |
| 2 | 0.774 | 0.672 | 0.810 | 0.851 | 0.822 | 0.864 | 0.861 | 0.837 |
| 3 | 0.686 | 0.672 | 0.737 | 0.787 | 0.738 | 0.787 | 0.801 | 0.786 |
| 4 | 0.623 | 0.672 | 0.630 | 0.691 | 0.661 | 0.686 | 0.699 | 0.676 |
| 5 | 0.611 | 0.672 | 0.675 | 0.715 | 0.693 | 0.714 | 0.721 | 0.719 |
| 6 | 0.651 | 0.672 | 0.730 | 0.794 | 0.750 | 0.789 | 0.786 | 0.782 |
| 7 | 0.631 | 0.672 | 0.702 | 0.754 | 0.723 | 0.750 | 0.749 | 0.757 |
| 8 | 0.698 | 0.672 | 0.731 | 0.756 | 0.744 | 0.761 | 0.758 | 0.763 |
| 9 | 0.598 | 0.672 | 0.669 | 0.659 | 0.649 | 0.691 | 0.693 | 0.695 |
| 10 | 0.706 | 0.724 | 0.799 | 0.833 | 0.789 | 0.834 | 0.843 | 0.841 |
| 11 | 0.632 | 0.654 | 0.653 | 0.715 | 0.703 | 0.709 | 0.732 | 0.726 |
| 12 | 0.609 | 0.614 | 0.639 | 0.732 | 0.665 | 0.708 | 0.721 | 0.723 |
| 13 | 0.493 | 0.490 | 0.531 | 0.653 | 0.556 | 0.580 | 0.625 | 0.630 |
| 14 | 0.598 | 0.588 | 0.641 | 0.666 | 0.658 | 0.695 | 0.689 | 0.707 |
| 15 | 0.561 | 0.575 | 0.636 | 0.707 | 0.622 | 0.688 | 0.699 | 0.700 |
| 16 | 0.652 | 0.636 | 0.661 | 0.734 | 0.684 | 0.729 | 0.747 | 0.729 |
| 17 | 0.672 | 0.727 | 0.729 | 0.768 | 0.738 | 0.763 | 0.770 | 0.781 |
| 18 | 0.576 | 0.585 | 0.650 | 0.708 | 0.671 | 0.684 | 0.703 | 0.703 |
| 19 | 0.529 | 0.537 | 0.594 | 0.722 | 0.625 | 0.685 | 0.701 | 0.707 |

* WU model results are grabbed from the predictions by [Wu](https://github.com/xinwucwp/faultSeg/tree/master/data/validation/predict). 


* VNet is the original VNet implementation from [here](https://github.com/mattmacy/vnet.pytorch). The only change from VNet-original is all batchnorm is replaced with InstanceNorm and ReLUs with LeakyReLU. The model is trained with cross-entropy (CE) + dyanmically weighted Dice loss and outperforms the same model trained only with CE or Dice. 

* Modified VNet is a modification made to Vnet's downsampling and upsampling block and outperforms the original VNet and Wu. Unfortunately the modifications means, we are not very faithful to the original VNet model layout. This model is trained with pure CE loss. Not tested CE+Dice(dynamic weighting) loss for this model.


* UNet-Res1 is custom 3D UNet with Residual Blocks. The Res-block  is based on the Kaggle 2017 Data Science Bowl 2nd place winner, but has beedn modified to follow more closely a standard ResNet's Res-block layout. The model is trained with CE and outperforms all the previous models on the test set.

* UNet-Res2 **( code not released)** This is a custom 3D UNet with additional tricks in the model architecture. This model is the **SOTA** in this repo. Arxiv paper is in the works for this and the code will be released with that. TTA results are included for this model to show additional uplift with TTA. Again combo loss improves compared to training with CE loss only. 

* UNet-DS is a 3d UNet trained with Deep supervision based on a simplified version of [this](https://arxiv.org/abs/1903.09097). The model is very robust and adapted from a brain segmentation challenge. Unfortunetaly I cannot recall which one so I can't cite it properly. This is a bigger model and we are under utilizing the model capacity here by training with small set and no augmentations. This model is also **SOTA** here.

* Curios thing: Training with standard class weighting consistently produced much thicker faults than desired. Thus we switched to a combo-loss for the weaker models (VNet) where the weihting is done dynamically (or stochastic weighting) only for the Dice loss term. CE continues to be non weighted. 

* Models are trained on V100 AWS machines. For Unet-Res2 model memory requirement is on the high side. All models are trained with batch-size=num_gpu_cards_on_machine 

### Directory layout

* models : contains the model scripts
* loaders: simple custom loader scripts to efficiently feed the data (TODO:augmentations)
* scripts: to run the trainining (train.py) and prediction (predict.py) scripts. Both only work on GPU enabled devices.
* zoo : trained model files. 
* data: training and test data (is incomplete, grab is from Wu) in npy format.

### Training and Test Data

* Download training and test dataset from data dir. All data is in numpy format with shape X,Y,Z. The TestSet images have been used to benchmark the models. 

* Note the TestSet is incomplete in my link, so get it from [Wu](https://github.com/xinwucwp/faultSeg/tree/master/data/validation/seis). Run the predict_dat.py sript for this
 
### Requirements
```
pip install -r requirements.txt
```
Then
```
pip3 install torch==1.1.0 torchvision==0.3.0
```

