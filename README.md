![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg) ![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

<img src='imgs/emma2portrait.gif' width=384>

# Diverse Image-to-Image Translation via Disentangled Representations
[[Project Page]](http://vllab.ucmerced.edu/hylee/DRIT/)[[Paper]]()

Pytorch implementation for our diverse image-to-image translation method. With the proposed disentangled representation approach, we are able to produce diverse translation results without paired training images.

Contact: Hsin-Ying Lee (hlee246@ucmerced.edu) and Hung-Yu Tseng (htseng6@ucmerced.edu)

## Paper
Diverse Image-to-Image Translation via Disentangled Representations<br>
[Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/)\*, [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/)\*, [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Maneesh Kumar Singh](https://scholar.google.com/citations?user=hdQhiFgAAAAJ), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
European Conference on Computer Vision (ECCV), 2018 (**oral**) (* equal contribution)

Please cite our paper if you find it useful for your research.
```
@inproceedings{DRIT,
  author = {Lee, Hsin-Ying. and Tseng, Hung-Yu and Huang, Jia-Bin and Singh, Maneesh Kumar and Yang, Ming-Hsuan},
  booktitle = {European Conference on Computer Vision},
  title = {Diverse Image-to-Image Translation via Disentangled Representations},
  year = {2018}
}
```

## Example Results
<img src='imgs/teaser.png' width="1000px"/>

## Usage

### Prerequisites
- Python 3.5 or Python 3.6
- Pytorch 4.0 and torchvision (https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [Tensorflow](https://www.tensorflow.org/) (for tensorboard usage)
- We provide a Docker file for building the environment based on CUDA 9.0 and CuDNN 7.1.

### Install
- Clone this repo:
```
git clone https://github.com/HsinYingLee/DRIT.git
cd DRIT/src
```

## Datasets
- Download the dataset using the following script.
```
bash ./datasets/download_dataset.sh dataset_name
```
- Portrait: 6452 photography images from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 1811 painting images downloaded and cropped from [Wikiart](https://www.wikiart.org/).
- Cat2dog: 871 cat (birman) images, 1364 dog (husky, samoyed) images crawled and cropped from Google Images.
- You can follow the instructions in CycleGAN [website](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to download the Yosemite (winter, summer) dataset and artworks (monet, van Gogh) dataset. For photo <-> artrwork translation, we use the summer images in Yosemite dataset as the photo images.

## Training Examples
- Yosemite (summer <-> winter)
```
python3 train.py --dataroot ../datasets/yosemite -name yosemite
tensorboard --logdir ../logs/yosemite
```
Results and saved models can be found at `../results/yosemite`.

- Portrait (photograpy <-> painting)
```
python3 train.py --dataroot ../datasets/portrait --name portrait --concat 0
tensorboard --logdir ../logs/portrait
```
Results and saved models can be found at `../results/portrait`.

## Testing Example
- Download a pre-trained model
```
bash ./models/download_model.sh
```
- Generate results with randomly sampled attributes
  - Require folder `testA` (for a2b) or `testB` (for b2a) under dataroot
```
python3 test.py --dataroot ../datasets/yosemite --name yosemite_random --resume ../models/example.pth --a2b 0
```
Diverse generated summer images can be found at `../outputs/yosemite_random`

- Generate results with attributes encoded from given images
  - Require both folders `testA` and `testB` under dataroot
```
python3 test_transfer.py --dataroot ../datasets/yosemite --name yosemite_encoded --resume ../models/example.pth --a2b 0
```
Diverse generated summer images can be found at `../outputs/yosemite_encoded`

## Training options and tips
- Due to the usage of adaptive pooling for attribute encoders, our model supports various input size. For example, here's the result of Grayscale -> RGB using 340x340 images.
<img src='imgs/flower.png' width="900px"/>

- We provide two different methods for combining content representation and attribute vector. One is simple concatenation, the other is feature-wise transformation (learn to scale and bias features). In our experience, if the translation involves less shape variation (e.g. winter <-> summer), simple concatenation produces better results. On the other hand, for the translation with shape variation (e.g. cat <-> dog, photography <-> painting), feature-wise transformation should be used (i.e. set `--concat 0`) in order to generate diverse results.

- In our experience, using the multiscale discriminator often gets better results. You can set the number of scales manually with `--dis_scale`.

- There is a hyper-parameter "d_iter" which controls the training schedule of the content discriminator. The default value is d_iter = 3, yet the model can still generate diverse results with d_iter = 1. Set `--d_iter 1` if you would like to save some training time. 

- We also provide option `--dis_spectral_norm` for using spectral normalization (https://arxiv.org/abs/1802.05957). We use the code from the master branch of pytorch since pytorch 0.5.0 is not stable yet. However, despite using spectral normalization significantly stabilizes the training, we fail to observe consistent quality improvement. We encourage everyone to play around with various settings and explore better configurations.

- Since the log file will be large if you want to display the images on tensorboard, set `--no_img_display` if you like to display only the loss values.
