# Overview

These are diffusion models described in the paper [High Fidelity Visualization of What Your Self-Supervised Representation Knows About](https://arxiv.org/abs/2112.09164).
Included in this release are the following models:

 * Diffusion models trained at 128x128 resolution on ImageNet with the representations of SimCLR/Dino/VicReg/Barlow Twins and a supervised network.

# Datasets

All of the models we are releasing were trained on the [ILSVRC 2012 subset of ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)
Here, we describe characteristics of these datasets which impact model behavior:

**ILSVRC 2012 subset of ImageNet**: This dataset was curated in 2012 and consists of roughly one million images, each belonging to one of 1000 classes.
 * A large portion of the classes in this dataset are animals, plants, and other naturally-occurring objects.
 * Many images contain humans, although usually these humans aren’t reflected by the class label (e.g. the class “Tench, tinca tinca” contains many photos of people holding fish).

# Performance

These models are intended to generate samples consistent with their training distributions.
This has been measured in terms of FID, Precision, and Recall.
These metrics all rely on the representations of a [pre-trained Inception-V3 model](https://arxiv.org/abs/1512.00567),
which was trained on ImageNet, and so is likely to focus more on the ImageNet classes (such as animals) than on other visual features (such as human faces).

Qualitatively, the samples produced by these models often look highly realistic, especially when a diffusion model is combined with a noisy classifier.

# Intended Use

These models are intended to be used for research purposes only.
In particular, they can be used as a baseline for generative modeling research, or as a starting point to build off of for such research.
These models are not intended to be commercially deployed.
Additionally, they are not intended to be used to create propaganda or offensive imagery.

# Limitations

These models sometimes produce highly unrealistic outputs, particularly when generating images containing human faces.
This may stem from ImageNet's emphasis on non-human objects. Because ImageNet and LSUN contain images from the internet, they include photos of real people, and the model may have memorized some of the information contained in these photos.
However, these images are already publicly available, and existing generative models trained on ImageNet have not demonstrated significant leakage of this information.
