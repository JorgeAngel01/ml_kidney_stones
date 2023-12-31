import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import os
from torchvision import transforms, utils, datasets
import random

class KidneyImagesLoader(pl.LightningDataModule):
  """
  Class to pull the zip file from google drive and returns the data loaders
  for the training, validation and tests sets. It receives as input the path of
  the zip file, the percentage of the training set that is going to be used for validation
  and the transformations to be applied to the train/val sets and the test set.
  """
  def __init__(self,
               zip_path=None,
               images_path=None,
               val_percentage=0.2,
               train_batch_size = 16,
               train_transformations=[],
               test_transformations=[],
              seed=None):
    super().__init__()
    self.num_workers = 8
    if seed != None:
      random.seed(seed)
      np.random.seed(seed)
      pl.seed_everything(seed)
      #torch.manual_seed(seed)
      # if you are suing GPU
      #torch.cuda.manual_seed(seed)
      #torch.cuda.manual_seed_all(seed)
      #torch.backends.cudnn.enabled = False 
      #torch.backends.cudnn.benchmark = False
      #torch.backends.cudnn.deterministic = True
      # self.num_workers = 0
      
    if not zip_path and not images_path:
      raise Exception('Zip path or images path is required')
    
    if zip_path:
      self.zip_path = zip_path
      self.image_path = self.zip_path[self.zip_path.rindex("/"):]
      self.image_path = "/content" + self.image_path.replace(".zip", "")
    else:
      self.image_path = images_path

    if images_path:
      self.image_path = images_path
    
    self.val_percentage = val_percentage
    self.train_batch_size = train_batch_size
    self.train_transformations = train_transformations
    self.test_transformations = test_transformations

  """
  Returns the classes found.
  """
  def get_class_indices(self):
    return self.idx2class

  """
  Returns the training dataset and the mapping for index to class name.
  """
  def get_train_dataset(self):
    dataset = datasets.ImageFolder(root=self.image_path + "/train", transform = self.train_transformations)
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    self.idx2class = idx2class
    self.train_dataset = dataset
    return dataset, idx2class

  """
  Splits the training set into train and validation sets and it returns the
  generated SubsetRandomSamplers.
  """
  def _train_val_samplers(self):
    rps_dataset_size = len(self.train_dataset)
    rps_dataset_indices = list(range(rps_dataset_size))
    np.random.shuffle(rps_dataset_indices)

    val_split_index = int(np.floor(self.val_percentage * rps_dataset_size))
    train_idx, val_idx = rps_dataset_indices[val_split_index:], rps_dataset_indices[:val_split_index]

    self.train_sampler = SubsetRandomSampler(train_idx)
    self.val_sampler = SubsetRandomSampler(val_idx)

  """
  Returns the test dataset.
  """
  def get_test_dataset(self):
    dataset_test = datasets.ImageFolder(root = self.image_path + "/test",
                                          transform = self.test_transformations)
    return dataset_test

  """
  Executed automatically by lightning when a dataloader is requested.
  """
  def setup(self, stage=None):
    if hasattr(self, 'zip_path'):
      if not os.path.exists(self.image_path):
          cmd = 'yes | cp "'+  self.zip_path + '" -d "/content/"'
          os.system(cmd)
          cmd = 'unzip -o --qq ' + self.image_path + ".zip" + ' -d "' + self.image_path + '"'
          os.system(cmd)
    self.get_train_dataset()
    self._train_val_samplers()


  """
  Returns the train data loader.
  """
  def train_dataloader(self):
    return DataLoader(dataset=self.train_dataset, shuffle=False, batch_size=self.train_batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

  """
  Returns the validation data loader.
  """
  def val_dataloader(self):
    return DataLoader(dataset=self.train_dataset, shuffle=False, batch_size=1, sampler=self.val_sampler, num_workers=self.num_workers)

  """
  Returns the test data loader.
  """
  def test_dataloader(self):
    dataset = self.get_test_dataset()
    return DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=self.num_workers)
