# Code Implementation of TSFE

> Refining the Unseen: Self-supervised Two-stream Feature Extraction for Image Quality Assessment


## Requirements:

- numpy==1.23.4
- scipy==1.9.3
- Pillow==9.3.0
- scikit-image==0.19.3
- opencv-python==4.6.0.66
- vit-pytorch==0.40.2
- scikit-learn==1.2.0
- torch==1.13.1
- torchvision==0.14.1



## Useage:

### Calculate indicator

Use `cal_indicator.py` to calculate indicator like SSIM, FSIM and GMSD.


### Image Fidelity Stream

Execute the following code for the image fidelity stream to extract the image fidelity feature representation vector with contrastive learning framework and CNN-based encoder. 

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python fidelity.py 
        --epochs 50 
        --nprocs 8 
        --batch_size 512 
        --lr 0.006 
        --encoder_network "VGG16" 
        --model_path "./results/models/VGG16_amp" 
        --loss_path "./results/loss/VGG16_amp" 
        > ./fidelity.log 2>&1 &
```


### Image Structure Stream

Execute the following code for the image structure stream to extract the image structure feature representation vector with Vision Transformer.

```
nohup python structure.py 
        --model_path "./results/models/structure" 
        --loss_path "./results/loss/structure" 
        --device "cuda:0" 
        > ./structure.log 2>&1 &
```


### Regression and Evaluation

Use `regression.py` to apply ridge regressor and evaluate the performance.

## Citation

```
@INPROCEEDINGS{10415683,
  author={Lou, Yiwei and Chen, Yanyuan and Xu, Dexuan and Zhou, Doudou and Cao, Yongzhi and Wang, Hanpin and Huang, Yu},
  booktitle={2023 IEEE International Conference on Data Mining (ICDM)}, 
  title={Refining the Unseen: Self-supervised Two-stream Feature Extraction for Image Quality Assessment}, 
  year={2023},
  pages={1193-1198},
  doi={10.1109/ICDM58522.2023.00147}
}
```
