# SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers

Spatial Autocorrelation Token Analysis (SATA) enhances the robustness and efficiency of Vision Transformers (ViTs) without retraining. While previous efforts to improve ViTs relied on heavy training strategies, augmentation, or structural changes, SATA leverages spatial relationships between token features. By grouping tokens based on spatial autocorrelation scores before the Feed-Forward Network (FFN) block, it boosts both representational power and computational efficiency. SATA integrates seamlessly with pre-trained ViTs and achieves state-of-the-art results on **ImageNet-1K (94.9% top-1)** and robustness benchmarks like **ImageNet-A (63.6%)**, **ImageNet-R (79.2%)**, and **ImageNet-C (mCE 13.6%)**— **all without additional training or fine-tuning**.


## SATA pipeline
### Geographical Spatial Auto-correlation

![image](https://github.com/user-attachments/assets/19338553-3faa-4494-9c80-60377ca0645b)

![sata-pipeline](https://github.com/user-attachments/assets/6123c376-9dde-4819-96ad-d0a8a5c7df70)

### ViTs' Robustness Performance
![robust](https://github.com/user-attachments/assets/18b6d080-f71c-4b87-a8f0-d7482d9572e6)

### ImageNet-1k Classification Performance
![imagenet-1k](https://github.com/user-attachments/assets/67f71566-7eb0-4c0e-89ff-5ebc788338a2)

### Run
1- Ensure that all essential libraries and packages (pytorch, timm, and thop) are installed.

2- Run the **main-val.py** file using the following commands and settings:

```bash
python main-eval.py --model_name "deit_base_patch16_224" --gamma 0.7 --data_path ./ImageNet2012/val/ --sata 
```

### References
1. Nikzad, Nick, Yi Liao,Yongsheng Gao, and Jun Zhou. "SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers", Accepted by Computer Vision and Pattern Recognition Conference (CVPR). 2025.
2. Nikzad, Nick, Yongsheng Gao, and Jun Zhou. ["CSA-Net: Channel-wise Spatially Autocorrelated Attention Networks."](https://arxiv.org/abs/2405.05755) arXiv preprint arXiv:2405.05755. 2024.
