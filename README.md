# ReColorAdv

This is an implementation of the ReColorAdv adversarial attack and other attacks described in the NeurIPS 2019 paper ["Functional Adversarial Attacks"](https://arxiv.org/abs/1906.00001).

## Getting Started

Clone this repository by running

    git clone https://github.com/cassidylaidlaw/ReColorAdv

You can experiment with the ReColorAdv attack, by itself and combined with other attacks, in the [`getting_started.ipynb`](getting_started.ipynb) Jupyter notebook. You can also open the notebook in Google Colab via the badge below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cassidylaidlaw/ReColorAdv/blob/master/getting_started_colab.ipynb)

You can also install the ReColorAdv package with pip by running

    pip install recoloradv

## Evaluation Script (CIFAR-10)

The script [`evaluate_cifar10.py`](recoloradv/examples/evaluate_cifar10.py) will evaluate a model trained on CIFAR-10 against the adversarial attacks in Table 1 of the paper. For instance, to evaluate a CIFAR-10 model trained on delta (L-infinity) attacks against a ReColorAdv+delta attack, run

    python recoloradv/examples/evaluate_cifar10.py --checkpoint pretrained_models/delta.resnet32.pt --attack recoloradv+delta

## Evaluation Script (ImageNet)

The script [`evaluate_imagenet.py`](recoloradv/examples/evaluate_imagenet.py) will download a ResNet-50 trained on ImageNet and evaluate it against the ReColorAdv attack:

    python recoloradv/examples/evaluate_imagenet.py --imagenet_path /path/to/ILSVRC2012 --batch_size 50

## Citation

If you find this repository useful for your research, please cite our paper as follows:

    @inproceedings{laidlaw2019functional,
      title={Functional Adversarial Attacks},
      author={Laidlaw, Cassidy and Feizi, Soheil},
      booktitle={NeurIPS},
      year={2019}
    }

## Contact

For questions about the paper or code, please contact claidlaw@umd.edu.
