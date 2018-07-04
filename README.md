gif file

# Diverse Image-to-Image Translation via Disentangled Representations
[[Project Page]]()[[Paper]]()

Pytorch implementation for our diverse image-to-image translation method. With the proposed disentangled representation aproach, we are able to produce diverse translation results without paired training images.

Contact: Hsin-Ying Lee (hlee246@ucmerced.edu) and Hung-Yu Tseng (htseng6@ucmerced.edu)

## Paper
Diverse Image-to-Image Translation via Disentangled Representations:<br>
[Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/)\*, [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/)\*, [Maneesh Kumar Singh](https://scholar.google.com/citations?user=hdQhiFgAAAAJ), Huang](https://filebox.ece.vt.edu/~jbhuang/), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/):<br>
European Conference on Computer Vision (ECCV), 2018 (* indicates equal contribution)

Please cite our paper if you find it useful for your research.
```
@inproceedings{Lee_drit_2018,
  author = {H.-Y. Lee and H.-Y. Tseng and M. Kumar and J.-B. Huang and M.-H. Yang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  title = {Diverse Image-to-Image Translation via Disentangled Representations},
  year = {2018}
}
```

## Example Results

## Usage

### Prerequisites
- Pytorch 4.0
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- Tensorflow (for tensorboard)
- We provide a Docker file for building the environment based on CUDA 9.0 and CuDNN 7.1.

### Install
- Clone this repo:
```
git clone ...
cd DRIT/src
```

## Dataset

## Training Examples
- Yosemite summer <-> winter translation
```
python3 train.py --dataroot ../datasets/yosemite --concat 1 --name yosemite
tensorboard --logdir ../logs yosemite
```
Results and saved models can be found at `../results/yosemite`.

- cats <-> dogs translation
```
python3 train.py --dataroot ../datasets/yosemite --name cat2dog
tensorboard --logdir ../logs cat2dog
```
Results and saved models can be found at `../results/cat2dog`.

## Testing Example
- Download a pre-trained model
- Generate results in domain B from domain A
```
python3 test.py --dataroot ../datasets/yosemite --a2b 1 --random_z 1 --name yosemite --concat 1 --resume ../models/example.pth
```
Results can be found at `../outputs/cat2dog`.
