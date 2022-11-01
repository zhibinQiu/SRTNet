[Apache License 2.0](https://github.com/zhibinQiu/SRTNet/LICENSE)
## SRTNet: Time Domain Speech Enhancement Via Stochastic Refinement

### Training
Before you start training, you'll need to prepare a training dataset. The default dataset is VOICEBANK-DEMAND dataset. You can download them from [VOICEBANK-DEMAND](https://doi.org/10.7488/ds/2117)). By default, this implementation assumes a sample rate of 16 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

You need to set the output path and data path under path.sh

```
output_path=[path_to_output_directory]
voicebank=[path_to_voicebank_directory]
```

Usage:
Train SE model
```
./train.sh [stage] [model_directory]
```


#### Multi-GPU training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.

### Validatoin and Inference API

Usage:
```
./valid.sh [stage] [model name] [checkpoint id] 
./inference.sh [stage] [model name] [checkpoint id]
```
## References
- [SRTNet: Time Domain Speech Enhancement Via Stochastic Refinement](https://arxiv.org/abs/2210.16805)
- [Conditional Diffusion Probabilistic Model for Speech Enhancement](https://arxiv.org/abs/2202.05256)
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
