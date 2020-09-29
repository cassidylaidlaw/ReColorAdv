
import torch
import argparse
import sys
import os
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.datasets import ImageNet
from torchvision import transforms

# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap 
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st
from recoloradv.mister_ed.utils import pytorch_utils as utils
from recoloradv.mister_ed.cifar10 import cifar_loader

# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs
from recoloradv.utils import get_attack_from_name, load_pretrained_cifar10_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a model trained on CIFAR-10 '
        'against ReColorAdv and other attacks'
    )

    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint to evaluate')
    parser.add_argument('--attack', type=str,
                        help='attack to run, such as "recoloradv" or '
                        '"stadv+delta"')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    args = parser.parse_args()

    model, normalizer = load_pretrained_cifar10_model(args.checkpoint)
    val_loader = cifar_loader.load_cifar_data('val', batch_size=args.batch_size)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    attack = get_attack_from_name(args.attack, model, normalizer)

    batches_correct = []
    for batch_index, (inputs, labels) in enumerate(val_loader):
        if (
            args.num_batches is not None and
            batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        adv_inputs = attack.attack(
            inputs,
            labels,
        )[0]
        with torch.no_grad():
            adv_logits = model(normalizer(adv_inputs))
        batch_correct = (adv_logits.argmax(1) == labels).detach()

        batch_accuracy = batch_correct.float().mean().item()
        print(f'BATCH {batch_index:05d}',
              f'accuracy = {batch_accuracy * 100:.1f}',
              sep='\t')
        batches_correct.append(batch_correct)

    accuracy = torch.cat(batches_correct).float().mean().item()
    print('OVERALL    ',
          f'accuracy = {accuracy * 100:.1f}',
          sep='\t')
