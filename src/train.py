import torch
from torch.autograd import Variable
from itertools import izip
from options.train_options import TrainOptions
from dataset_unpair import dataset_unpair
from model import DRIT
from saver import Saver

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  dataset_a = dataset_unpair(opts, opts.listA, opts.input_dim_a)
  dataset_b = dataset_unpair(opts, opts.listB, opts.input_dim_b)
  train_loader_a = torch.utils.data.DataLoader(dataset_a, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
  train_loader_b = torch.utils.data.DataLoader(dataset_b, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
  print('\n--- load dataset ---')
  print('A: %d, B: %d images'%(len(dataset_a), len(dataset_b)))

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
  else:
    ep0 = model.resume(opts.resume)
  model.setgpu(opts.gpu)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  max_it = 500000
  total_it = 0
  for ep in range(ep0, opts.n_ep):
    for it, (images_a, images_b) in enumerate(izip(train_loader_a, train_loader_b)):
      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue

      # input data
      images_a = Variable(images_a.cuda(opts.gpu))
      images_b = Variable(images_b.cuda(opts.gpu))

      # update model
      if (it + 1) % opts.d_iter != 0:
        model.update_D_content(images_a, images_b)
        continue
      else:
        model.update_D(images_a, images_b)
        model.update_EG()

      # decay learn rate
      if opts.n_ep_decay > -1:
        model.update_lr()

      # save to display file
      saver.write_display(total_it, model)


      print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-41608, model)
        saver.write_model(-41608, model)
        break

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, model)

  return

if __name__ == '__main__':
  main()
