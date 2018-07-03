import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT, DRIT_concat
from saver import save_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  if opts.a2b:
    dataset = dataset_single(opts, 'A', opts.input_dim_a)
  else:
    dataset = dataset_single(opts, 'B', opts.input_dim_b)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  if opts.concat:
    model = DRIT_concat(opts)
  else:
    model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  for idx1, img in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img = img.cuda()
    imgs = [img]
    names = ['input']
    with torch.no_grad():
      imgs_list = model.interpolate(img, 'gg/6.npy', 'gg/4.npy', a2b=opts.a2b)
    for idx2 in range(len(imgs_list)):
      imgs.append(imgs_list[idx2])
      names.append('output_{}'.format(idx2))
    save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(idx1)))

  return

if __name__ == '__main__':
  main()
