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
```


## Usage
```bash
# Train 
$ cd LatentComposer
$ CUDA_VISIBLE_DEVICES=0 python main.py config/composer.yaml -t --gpus 0, 

```


## Implementation Notes
I implemented the DDIM function(eq. 3 in the paper) as below. Since this part is not clear to me, I am not totally sure, if this is the right implementation or not. So I would appreaciate if someone give a hint for this point.

```python
def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
b, *_, device = *x.shape, x.device
model_c1, model_c2 = self.model.apply_model(x, t, c).chunk(2)
model_output = model_c1 + guidance_scale * (model_c2 - model_c1) 
...
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
