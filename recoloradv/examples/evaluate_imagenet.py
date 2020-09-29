
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

# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a ResNet-50 trained on Imagenet '
        'against ReColorAdv'
    )

    parser.add_argument('--imagenet_path', type=str, required=True,
                        help='path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    args = parser.parse_args()

    model = resnet50(pretrained=True, progress=True)
    normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    dataset = ImageNet(
        args.imagenet_path,
        split='val',
        transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    cw_loss = lf.CWLossF6(model, normalizer, kappa=float('inf'))
    perturbation_loss = lf.PerturbationNormLoss(lp=2)
    adv_loss = lf.RegularizedLoss(
        {'cw': cw_loss, 'pert': perturbation_loss},
        {'cw': 1.0, 'pert': 0.05},
        negate=True,
    )

    pgd_attack = aa.PGD(
        model,
        normalizer,
        ap.ThreatModel(pt.ReColorAdv, {
            'xform_class': ct.FullSpatial,
            'cspace': cs.CIELUVColorSpace(),
            'lp_style': 'inf',
            'lp_bound': 0.06,
            'xform_params': {
              'resolution_x': 16,
              'resolution_y': 32,
              'resolution_z': 32,
            },
            'use_smooth_loss': True,
        }),
        adv_loss,
    )

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

        adv_inputs = pgd_attack.attack(
            inputs,
            labels,
            optimizer=optim.Adam,
            optimizer_kwargs={'lr': 0.001},
            signed=False,
            verbose=False,
            num_iterations=(100, 300),
        ).adversarial_tensors()
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
