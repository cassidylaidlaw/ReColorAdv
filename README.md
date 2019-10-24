# ReColorAdv

This is an implementation of the ReColorAdv adversarial attack described in the NeurIPS 2019 paper ["Functional Adversarial Attacks"](https://arxiv.org/abs/1906.00001).

## Getting Started

ReColorAdv depends on the [`mister_ed`](https://github.com/revbucket/mister_ed) library. To install, navigate to the `ReColorAdv` directory and run

    git clone https://github.com/revbucket/mister_ed.git
    cd mister_ed
    pip install -r requirements.txt
    python scripts/setup_cifar.py

This will also download the CIFAR-10 dataset and pretrained models for it. Once `mister_ed` is installed, you can experiment with the ReColorAdv attack, by itself and combined with other attacks, in the `getting_started.ipynb` Jupyter notebook.

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
