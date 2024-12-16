# HINER: Neural Representation for Hyperspectral Image


## Overview
This is the official implementation of HINER (ACM MM'24), a novel neural representation for **compressing HSI** and **ensuring high-quality downstream tasks on compressed HSI**.

* **Compressing HSI:** HINER fully exploits inter-spectral correlations by **explicitly encoding of spectral wavelengths** and achieves a compact representation of the input HSI sample through joint optimization with a learnable decoder. By additionally incorporating the **Content Angle Mapper** with the L1 loss, we can supervise the global and local information within each spectral band, thereby enhancing the overall reconstruction quality.

* **Ensuring high-quality downstream tasks on compressed HSI:** For downstream classification on compressed HSI, we theoretically demonstrate the task accuracy is not only related to the classification loss but also to the reconstruction fidelity through a first-order expansion of the accuracy degradation, and accordingly adapt the reconstruction by introducing **Adaptive Spectral Weighting**. Owing to the monotonic mapping of HINER between wavelengths and spectral bands, we propose **Implicit Spectral Interpolation** for data augmentation by adding random variables to input wavelengths during classification model training.

* **Experiments:** Experimental results on various HSI datasets demonstrate the superior compression performance of our HINER compared to the existing learned methods and also the traditional codecs. Our model is lightweight and computationally efficient, which maintains high accuracy for downstream classification task even on decoded HSIs at high compression ratios.


## TODO
✅ HSI Compression (before 2025.1)

☑️ Classification on compressed HSI (before 2025.3)

☑️ More implementations for compared literatures (before 2025.3)

☑️ Optimize Quantization (before 2025.4)


## Requirement
```bash
pip install -r requirements.txt
```

## Quick Usage for Compression
1. **Train HINER in IndianPine**
```bash
CUDA_VISIBLE_DEVICES=0 python train_hiner.py  --outf HINER \
    --data_path data/IndianPine.mat --vid IndianPine.mat --data_type HSI \
    --arch hiner --conv_type none pshuffel --act gelu --norm none  --crop_list 180_180  --ori_shape 146_146_200 \
    --resize_list -1 --loss SAM  --enc_dim 64_16 \
    --quant_model_bit 8 --quant_embed_bit 8 --fc_hw 3_3 \
    --dec_strds 5 3 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.0  -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
```

2. **Train HINER in PaviaUniversity**
```bash
CUDA_VISIBLE_DEVICES=1 python train_hiner.py  --outf HINER \
    --data_path data/PaviaU.mat --vid PaviaU.mat --data_type HSI  \
    --arch hiner --conv_type convnext pshuffel --act gelu --norm none  --crop_list 720_360  --ori_shape 610_340_103 \
    --resize_list -1 --loss SAM  --enc_dim 64_16 \
    --quant_model_bit 8 --quant_embed_bit 8 --fc_hw 6_3\
    --dec_strds 5 4 3 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.0  -e 300 --eval_freq 30  --lower_width 12 -b 1 --lr 0.001
```

## Quick Usage for Classification on compressed HSI
```bash
    Coming Soon!
```


## Contact
Junqi Shi: junqishi@smail.nju.edu.cn

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```bash
@inproceedings{shi2024hiner,
  title={HINER: Neural Representation for Hyperspectral Image},
  author={Shi, Junqi and Jiang, Mingyi and Lu, Ming and Chen, Tong and Cao, Xun and Ma, Zhan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9837--9846},
  year={2024}
}
```

## Acknowledgement
This framework is based on [HNeRV](https://github.com/haochen-rye/HNeRV) and [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)

We thank the authors for sharing their codes.
