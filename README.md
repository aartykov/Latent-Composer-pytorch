# Latent-Composer-Pytorch (wip)

Implementation of a light-weighted Latent-Composer in PyTorch based on "Composer: Creative and Controllable Image Synthesis with Composable Conditions". 


## TODO

- [ ] Implement color palette decomposition.
- [ ] Reimplement the DDIM sampling code.
- [ ] Release inference code.
- [ ] Release pretrained models.


## Install
```bash
$ git clone https://github.com/aartykov/Latent-Composer-pytorch.git 
$ conda env create -f environment.yaml
$ conda activate latent-composer

- Download stable diffusion v1.5 checkpoints from https://huggingface.co/runwayml/stable-diffusion-v1-5 and put it inside the /models directory.
- Download MiDAS checkpoints from https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt and put it inside the /annotators/ckpts directory.
- Download quantized VAE autoencoder checkpoints from https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip and put it inside the /models/first_stage_models/kl-f8 directory.
```


## Usage
```bash
# Train 
$ cd Latent-Composer-pytorch
$ CUDA_VISIBLE_DEVICES=0 python main.py config/composer.yaml -t --gpus 0, 

```


## Implementation Notes

```python

```

## Citations

```bibtex
@article{lhhuang2023composer,
  title={Composer: Creative and Controllable Image Synthesis with Composable Conditions},
  author={Huang, Lianghua and Chen, Di and Liu, Yu and Yujun, Shen and Zhao, Deli and Jingren, Zhou},
  booktitle={arXiv preprint arxiv:2302.09778},
  year={2023}
}
```

```bibtex
@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
  author={Lvmin Zhang and Maneesh Agrawala},
  year={2023},
  eprint={2302.05543},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
