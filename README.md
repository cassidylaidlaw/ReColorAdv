# ReColorAdv

This is an implementation of the ReColorAdv adversarial attack described in the paper "Functional Adversarial Attacks."

## Getting Started

ReColorAdv depends on the `mister_ed` library. To install, navigate to the `ReColorAdv` directory and run

    git clone https://github.com/revbucket/mister_ed.git
    cd mister_ed
    pip install -r requirements.txt
    python scripts/setup_cifar.py

This will also download the CIFAR-10 dataset and pretrained models for it. Once `mister_ed` is installed, you can experiment with the ReColorAdv attack, by itself and combined with other attacks, in the `getting_started.ipynb` Jupyter notebook.
