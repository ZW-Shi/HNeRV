# HNeRV: A Hybrid Neural Representation for Videos  (CVPR 2023)
### [Paper](https://arxiv.org/abs/2304.02633) | [Project Page](https://haochen-rye.github.io/HNeRV) | [UVG Data](http://ultravideo.fi/#testsequences) 


[Hao Chen](https://haochen-rye.github.io),
Matthew Gwilliam,
Ser-Nam Lim,
[Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)<br>
This is the official implementation of the paper "HNeRV: A Hybrid Neural Representation for Videos".

## TODO 
- [ &check; ] Video inpainting
- [ &check; ] Fast loading from video checkpoints
- [ ] Upload results and checkpoints for UVG

## Method overview

<p float="left">
<img src="https://i.imgur.com/SdRcEiY.jpg"  height="190" />
<img src="https://i.imgur.com/CAppWSM.jpg"  height="190" />
</p>

## Get started
We run with Python 3.8, you can set up a conda environment with all dependencies like so:
```
pip install -r requirements.txt 
```

## High-Level structure
The code is organized as follows:
* [train_nerv_all.py](./train_nerv_all.py) includes a generic traiing routine.
* [model_all.py](./model_all.py) contains the dataloader and neural network architecure 
* [data/](./data) directory video/imae dataset, we provide bunny frames here
* [checkpoints/](./checkpoints) we provide model weights, and quantized video checkpoints for bunny here
* log files (tensorboard, txt, state_dict etc.) will be saved in output directory (specified by ```--outf```)
* We provide numerical results for distortion-compression at [uvg_results](./checkpoints/uvg_results.csv) and [per_video_results](./checkpoints/uvg_per_vid_results.csv) .


## Reproducing experiments

### Training HNeRV
HNeRV of 1.5M is specified with ```'--modelsize 1.5'```, and we balance parameters with ```'-ks 0_1_5 --reduce 1.2' ```
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

### NeRV baseline
NeRV baseline is specified with ```'--embed pe_1.25_80 --fc_hw 8_16'```, with imbalanced parameters ```'--ks 0_3_3 --reduce 2' ```
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
   --resize_list -1 --loss L2   --embed pe_1.25_80 --fc_hw 8_16 \
    --dec_strds 5 4 2 2 --ks 0_3_3 --reduce 2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

### Evaluation & dump images and videos
To evaluate pre-trained model, use ```'--eval_only --weight [CKT_PATH]'``` to evaluate and specify model path. \
For model and embedding quantization, use ```'--quant_model_bit 8 --quant_embed_bit 6'```.\
To dump images or videos, use  ```'--dump_images --dump_videos'```.
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2  \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001 \
   --eval_only --weight checkpoints/hnerv-1.5m-e300.pth \
   --quant_model_bit 8 --quant_embed_bit 6 \
    --dump_images --dump_videos
```

### Video inpainting
We can specified inpainting task with ```'--vid bunny_inpaint_50'``` where '50' is the mask size.
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny_inpaint_50   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

### Efficient video loading
We can load video efficiently from a [tiny checkpoint](./checkpoints/quant_vid.pth).\
Specify decoder and checkpoint by ```'--decoder [Decoder_path] --ckt [Video checkpoint]'```, output dir and frames by ```'--dump_dir [out_dir] --frames [frame_num]'```.
```
python efficient_nvloader.py --frames 16
```

## Neural Low-Rank Tensor Representation (NLRT)

NLRT replaces the intermediate video feature map with a neural CP low-rank tensor field:
\[
\mathcal{F}(t,h,w,c)=\sum_{r=1}^{R} A_\theta(t,r)\,B_\phi(h,r)\,D_\psi(w,r)\,E_\omega(c,r)
\]
where \(A_\theta\) is a temporal factor network (with temporal PE + MLP), \(B_\phi,D_\psi\) are spatial factors (parameter tables or optional spatial MLP), and \(E_\omega\) is a channel factor matrix.

### New arguments
- `--repr_type`: `baseline|nlrt` (default `baseline`)
- `--nlrt_rank`: CP rank `R` (default `0`, i.e. fallback to baseline path)
- `--nlrt_h0`, `--nlrt_w0`, `--nlrt_c`: NLRT field shape
- `--nlrt_t_pe`: temporal PE config (`base_levels`, e.g. `1.25_16`)
- `--nlrt_lambda_t`: temporal smoothness loss weight
- `--nlrt_lambda_reg`: factor regularization loss weight
- `--nlrt_use_spatial_mlp`: use spatial coordinate MLP (`0/1`)

### Minimal run commands
Baseline (original path):
```bash
python train_nerv_all.py --data_path data/bunny --vid bunny \
  --conv_type convnext pshuffel --act gelu --norm none \
  --crop_list 640_1280 --resize_list -1 --loss L2 \
  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
  --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2 \
  --modelsize 0.2 -e 1 --eval_freq 1 --lower_width 12 -b 1 --workers 0 --debug
```

NLRT:
```bash
python train_nerv_all.py --data_path data/bunny --vid bunny \
  --conv_type convnext pshuffel --act gelu --norm none \
  --crop_list 640_1280 --resize_list -1 --loss L2 \
  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
  --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2 \
  --modelsize 0.2 -e 1 --eval_freq 1 --lower_width 12 -b 1 --workers 0 --debug \
  --repr_type nlrt --nlrt_rank 16 --nlrt_h0 2 --nlrt_w0 4 --nlrt_c 16 \
  --nlrt_t_pe 1.25_16 --nlrt_lambda_t 1e-4 --nlrt_lambda_reg 1e-6
```

### One-click ablation
```bash
bash scripts/ablate_nlrt.sh
```
This writes per-run logs into `logs/` and an aggregated table into `results/nlrt_ablation.csv`.

### Reproducibility notes
- Training seeds are fixed for Python / NumPy / PyTorch.
- `cudnn.deterministic=True` and `cudnn.benchmark=False` are enabled for reproducibility (typically slower than non-deterministic settings).
- CSV outputs include environment info and Git commit hash.

### Example result table (see `results/nlrt_ablation.csv`)
| run_name | repr_type | nlrt_rank | pred_seen_psnr | pred_seen_ssim | lpips | param_count_m | checkpoint_size_mb | decode_ms_per_frame |
|---|---|---:|---:|---:|---|---:|---:|---:|
| baseline | baseline | 0 | (generated by script) | (generated by script) | N/A | ... | ... | ... |
| nlrt_r8 | nlrt | 8 | (generated by script) | (generated by script) | N/A | ... | ... | ... |
| nlrt_r16 | nlrt | 16 | (generated by script) | (generated by script) | N/A | ... | ... | ... |
| nlrt_r32 | nlrt | 32 | (generated by script) | (generated by script) | N/A | ... | ... | ... |

## Citation
If you find our work useful in your research, please cite:
```
@InProceedings{chen2023hnerv,
      title={{HN}e{RV}: Neural Representations for Videos}, 
      author={Hao Chen and Matthew Gwilliam and Ser-Nam Lim and Abhinav Shrivastava},
      year={2023},
      booktitle={CVPR},
}
```

## Contact
If you have any questions, please feel free to email the authors: chenh@umd.edu
