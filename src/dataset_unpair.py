import os
import torch
import cv2
import torch.utils.data as data
import numpy as np

class dataset_unpair(data.Dataset):
  def __init__(self, opts, listname, input_dim):
    self.dataroot = opts.dataroot
    list_fullpath = os.path.join(self.dataroot, listname)
    with open(list_fullpath) as f:
      content = f.readlines()
    f.close()
    self.images = [os.path.join(self.dataroot, 'data', x.strip().split(' ')[0]) for x in content]
    self.dataset_size = len(self.images)
    self.flip = not opts.no_flip
    self.image_size = opts.crop_size
    self.input_dim = input_dim
    return

  def __getitem__(self, index):
    crop_img = self.load_img(self.images[index], flip=self.flip, random_crop=True)
    raw_data = crop_img.transpose((2, 0, 1))
    data = ((torch.FloatTensor(raw_data) / 255.0) - 0.5) * 2
    return data

  def load_img(self, img_name, flip=False, random_crop=False):
    if self.input_dim == 3:
      img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
      h, w, c = img.shape
      img = np.float64(img)
    else:
      assert(self.input_dim == 1)
      img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      h, w = img.shape
      img = np.float64(img)
      img = np.expand_dims(img, axis=2)

    # flipping
    if flip:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)

    # cropping
    if random_crop:
      x_offset = np.int32(np.random.randint(0, w - self.image_size + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - self.image_size + 1, 1))[0]
    else:
      x_offset = np.int((w - self.image_size) / 2)
      y_offset = np.int((h - self.image_size) / 2)
    crop_img = img[y_offset:(y_offset + self.image_size), x_offset:(x_offset + self.image_size), :]
    return crop_img

  def __len__(self):
    return self.dataset_size
