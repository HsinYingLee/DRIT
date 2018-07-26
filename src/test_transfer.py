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
  datasetA = dataset_single(opts, 'A', opts.input_dim_a)
  datasetB = dataset_single(opts, 'B', opts.input_dim_b)
  if opts.a2b:
    loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads, shuffle=True)
  else:
    loader = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads, shuffle=True)

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
    for idx2, img2 in enumerate(loader_attr):
      if idx2 == opts.num:
        break
      img2 = img2.cuda()
      with torch.no_grad():
        if opts.a2b:
          img = model.test_forward_transfer(img1, img2, a2b=True)
        else:
          img = model.test_forward_transfer(img2, img1, a2b=False)
      imgs.append(img)
      names.append('output_{}'.format(idx2))
    save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(idx1)))

  return

if __name__ == '__main__':
  main()
