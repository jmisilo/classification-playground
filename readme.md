## Classification Playground

The goal of the project was to implement a very simple model for the classification of datasets from sklearn. [W&B](wandb.ai) was used to optimize hyperparameters. The model has been implemented using [PyTorch](https://pytorch.org/). I used datasets generated with `make_blobs` and `make_moons` methods. Both were trained at the same time, using only my Notebook with CPU. Times:

- **Moons** - 12min 28s
- **Blobs** - 11min 56s

Models' max accuracy on the validation sets:

- **Moons** - 99.33%
- **Blobs** - 91.33%

Detailed results of training process:

**SOON...**