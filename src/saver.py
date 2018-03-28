import os
import torchvision
import tensorboard
from tensorboard import summary
import numpy as np

class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.display_freq = opts.display_freq
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    if not os.path.exists(self.display_dir):
      os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)

    self.train_writer = tensorboard.FileWriter("%s" % (self.display_dir))

  def write_display(self, total_it, model):
    if (total_it + 1) % self.display_freq == 0:
      # write loss
      members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
      for m in members:
        self.train_writer.add_summary(summary.scalar(m, getattr(model, m)), total_it)
      # write img
      image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)/2, normalize=True, scale_each=True)
      image_dis = np.transpose(image_dis.numpy(), (1, 2, 0)) * 255
      image_dis = image_dis.astype('uint8')
      self.train_writer.add_summary(summary.image('Image', image_dis), total_it)

  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
      assembled_images = model.assemble_outputs3()
      img_filename = '%s/gen_%08d.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)
    elif ep == -41608:
      assembled_images = model.assemble_outputs3()
      img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)

  def write_model(self, ep, model):
    if (ep + 1) % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save('%s/%08d.pkl' % (self.model_dir, ep), ep)
    elif ep == -41608:
      model.save('%s/last.pkl' % self.model_dir, ep)

