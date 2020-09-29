# Pretrained models

This directory contains pretrained models for CIFAR-10 from the paper. All models use the ResNet-32 architecture in `recoloradv.mister_ed.cifar10.cifar_resnets`. The pretrained models include
 * A normally-trained model: [normal.resnet32.pt](normal.resnet32.pt).
 * A model trained on black-and-white images: [bw.resnet32.pt](bw.resnet32.pt).
 * Models trained on combinations of ReColorAdv, StAdv, and delta (L-infinity) attacks: [delta.resnet32.pt](delta.resnet32.pt), [stadv.resnet32.pt](stadv.resnet32.pt), [recoloradv.resnet32.pt](recoloradv.resnet32.pt), [stadv_delta.resnet32.pt](stadv_delta.resnet32.pt), [recoloradv_stadv.resnet32.pt](recoloradv_stadv.resnet32.pt), [recoloradv_delta.resnet32.pt](recoloradv_delta.resnet32.pt), and [recoloradv_stadv_delta.resnet32.pt](recoloradv_stadv_delta.resnet32.pt).


