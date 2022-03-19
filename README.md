# Continual Learning in Deep Neural Networks

<p align="justify"> The purpose of continual learning is to enable the network to train on sequence of datasets, without a significant decrease in the classification accuracy of each of them. Three methods of continual learning have been implemented: </p> 

- Elastic Weight Concolidation (EWC) [https://arxiv.org/abs/1612.00796](url)
- PackNet [https://arxiv.org/abs/1711.05769](url)
- Generative Replay [https://www.nature.com/articles/s41467-020-17866-2](url)

<p align="justify"> Python scripts have been created to test these methods using MLP or CNN as a base model. Coninual learning experiments are created using <b>MNIST, Fashion-MNIST and CIFAR-100 datasets.</b> The considered algorithms are able to prevent catastrophic forgetting in the scenarios of incremental task learning, domain adaptation and incremental class learning. Generative Replay was also modified by using GANs for image generation, which significantly improves the results on the CIFAR-100 dataset. </p>

## Requirements
```python
pip install -r requirements.txt
```
- `torch==1.9.0+cu102`
- `torchvision==0.10.0+cu102`
- `numpy`
- `matplotlib`
- `prettytable`

## Runnnig experiments
Scripts `EWC.py`, `PackNet.py`, `GR_vae.py` and `GR_gan.py` perform training and testing with individual algorithms. In Generative Replay method, 2 types of generators are available: Variational Autoencoder (VAE) or Generative Adversarial Networks (GAN).

To compare all 3 algorithms, as well as base model without CL capabilities, run:
```python
python compare.py --ewc --packnet --gen_rep --base_model
```

Parser of input arguments is defined in file `options.py` 

## Results
- Split MNIST - 5 binary classification tasks, in each task images of 2 types of digits

![ALL_splitMNIST_MLP_lr0 001_bs128_epochs10_pretrain10_avg](https://user-images.githubusercontent.com/92218640/138709936-cd91809a-7267-46e8-8502-2a6620bf3c0f.png)

---

- Permuted MNIST - training on sequence of 20 tasks, each task created by random permutation of original images pixels

![ALL_permMNIST_domain_20tasks_MLP_fc2000_lr0 001_bs128_epochs5_avg_task_end](https://user-images.githubusercontent.com/92218640/138709586-d4bdd95e-bb6a-40e0-96b2-d518f807b736.png)
