import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  if opts.a2b:
    dataset = dataset_single(opts, 'A', opts.input_dim_a)
  else:
    dataset = dataset_single(opts, 'B', opts.input_dim_b)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  for idx1, img1 in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img1 = img1.cuda()
    imgs = [img1]
    names = ['input']
    for idx2 in range(opts.num):
      with torch.no_grad():
        img = model.test_forward(img1, a2b=opts.a2b)
      imgs.append(img)
      names.append('output_{}'.format(idx2))
    save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(idx1)))

  return

if __name__ == '__main__':
  main()
