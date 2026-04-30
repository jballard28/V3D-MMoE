import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchio as tio
import json
from densenet import DenseNet
from torchvision.transforms import Compose, ToTensor, Normalize
from random import sample
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, orig_img_shapes, use_common_ids=True, rebalance=False):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.use_common_ids = use_common_ids
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = self.labels[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]
        self.orig_img_shapes = orig_img_shapes

        if rebalance:
            # Rebalance the dataset based on labels via resampling
            classes = set(self.label_new)
            labelcts = {c:0 for c in classes}
            for ll in self.label_new:
                labelcts[ll] += 1
            new_l = max(list(labelcts.values()))
            newids = {}
            newdata = {m:{} for m in self.data_new}
            newlabel = {}
            newmc = {}
            newobserved = {}
            # sample from each class to produce class balance
            for c in classes:
                class_idcs = list(np.where(np.array(self.label_new) == c)[0])
                sample_idcs = sample(class_idcs, min(len(class_idcs), new_l-len(class_idcs)))
                newids[c] = np.array(self.ids)[sample_idcs]
                for m in self.data_new:
                    newdata[m][c] = self.data_new[m][sample_idcs]
                newlabel[c] = np.array(self.label_new)[sample_idcs]
                newmc[c] = np.array(self.mc_new)[sample_idcs]
                newobserved[c] = self.observed_new[sample_idcs]
    
            # append new samples
            for m in newdata:
                newdata[m] = np.concatenate(list(newdata[m].values()))
                self.data_new[m] = np.concatenate((self.data_new[m],newdata[m]))
            for c in classes:
                self.ids = np.concatenate((self.ids, newids[c]))
                self.label_new = np.concatenate((self.label_new, newlabel[c]))
                self.mc_new = np.concatenate((self.mc_new, newmc[c]))
                self.observed_new = np.concatenate((self.observed_new, newobserved[c]), axis=0)

        # Sort ids by the number of available modalities
        self.sorted_ids = sorted(np.arange(len(self.ids)), key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), reverse=True)
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.label_new = self.label_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data_new.items():
            sample_data[modality] = data[idx]
            if modality == 'mri' or modality == 'fdg':
                subj1 = data[idx]
#                subj_gm_3d = np.zeros(self.masks.shape, dtype=np.float32)
#                subj_gm_3d.ravel()[self.masks] = subj1
#                subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
                subj_gm_3d = subj1.reshape(self.orig_img_shapes[modality])
                if self.transforms:
                    subj_gm_3d = self.transforms(subj_gm_3d)
                    sample = subj_gm_3d # Don't need to add channel dimension since this was done within the transform
                else:
                    sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
                sample_data[modality] = np.array(sample)

        label = self.label_new[idx]
        mc = self.mc_new[idx]
        observed = self.observed_new[idx]

        return sample_data, label, mc, observed

class ToTensor3D(torch.nn.Module):  
  def __init__(self):
    super().__init__()
  
  def forward(self, tensor):
#    y_new = torch.from_numpy(tensor.transpose(3,2,0,1))
    tensor = tensor[np.newaxis,:,:,:]
    y_new = torch.from_numpy(tensor)
    return y_new

  def __repr__(self):
    return self.__class__.__name__ + '()'
  
def DefineRangeNormalization(normalizzazione_range='range_intra', supLim=1, infLim=0):
  min_p, max_p = 0.0, 0.0
  
  if normalizzazione_range == 'range_intra':
    #supLim, infLim, type_n, max_p, min_p
    norm_range = NormalizeInRange(supLim, infLim, normalizzazione_range, 0, 0)
  else:
    norm_range = NormalizeInRange(0, 0, normalizzazione_range, 0, 0)
  
  return norm_range, min_p, max_p
  
class NormalizeInRange(torch.nn.Module):
  def __init__(self, supLim, infLim, type_n, max_p, min_p):
    self.supLim = supLim
    self.infLim = infLim
    self.type_n = type_n
    self.max_p = max_p
    self.min_p = min_p
    super().__init__()
    
  def forward(self, img):
    if self.type_n == 'range_intra':
      if torch.max(img) == torch.min(img):
          x_norm = img
      else:
          x_norm = ( (img - torch.min(img)) / (torch.max(img)- torch.min(img)) )*(self.supLim - self.infLim) + self.infLim
          assert torch.min(x_norm) >= self.infLim
          assert torch.max(x_norm) <= self.supLim
      
    elif self.type_n == 'range_inter':
      x_norm = ( (img - self.min_p) / (self.max_p- self.min_p) )*(self.supLim - self.infLim) + self.infLim
      assert torch.min(x_norm) >= self.infLim
      assert torch.max(x_norm) <= self.supLim
    
    else: 
      x_norm = img
 
    return x_norm
    
  def __repr__(self):
    return self.__class__.__name__ + '(supLim={}, infLim = {}, type_n = {}, max_p = {}, min_p = {})'.format(self.supLim, self.infLim, self.type_n, self.max_p, self.min_p)
          
class Resize3D(torch.nn.Module):  
  def __init__(self, size=(32,32,32), enable_zoom = False):
    self.size = size    
    self.enable_zoom = enable_zoom    
    super().__init__()         

  def forward(self, tensor):
    if self.enable_zoom:
#      img = F.interpolate( tensor.unsqueeze(0).unsqueeze(0), self.size, align_corners =True, mode='trilinear').squeeze(0).squeeze(0)
      img = F.interpolate( tensor.unsqueeze(0), self.size, align_corners =True, mode='trilinear').squeeze(0)
    else: 
      img = tensor
    return img
  
  def __repr__(self):
    return self.__class__.__name__ + '(size={}, enable_zoom = {})'.format(self.size, self.enable_zoom)
     
def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]

def load_and_preprocess_image_data(mod, data_dir, image_path, label_df, id_to_idx, mask_path = None):
    # Load and preprocess image data
    fname = mod+'_longit_m061224'
    hf = h5py.File(os.path.join(image_path, fname+'.hdf5'), 'r')
    image_data = np.array(hf[fname][:])
    orig_shape = image_data.shape[1:]

    # Flatten each image into a 1D array
    image_data = image_data.reshape(image_data.shape[0],-1)

    df = pd.read_csv(f'{data_dir}/adni/meta/longit_m061224_DX.csv')
    df = df.dropna(subset=['imgid_'+mod])
    df = df.reset_index(drop=True)

#    df = df.sort_values(by='Month', ascending=False)
#    idx = df.groupby('PTID')['Month'].idxmax()

    # Creating the subset DataFrame using the indexes
#    subdf = df.loc[idx]
    subdf = df[df['Month'] == 0.0]
    subdf = subdf.sort_index()
    subdf = subdf.reset_index()

    merged_df = subdf

    image_data = image_data[merged_df['index']]
    final_subject_ids = list(subdf.PTID)

    new_idx = np.array(convert_ids_to_index(final_subject_ids, id_to_idx))
    filtered_idx = [x for x in new_idx if x != -1]
    tmp = np.zeros((len(id_to_idx), image_data.shape[1])) - 2
    tmp[filtered_idx] = image_data[np.array(new_idx) != -1]

    mask_gm = None
    if mask_path is not None:
        data = nib.load(mask_path).get_fdata()
        mask_gm = (data == 150).ravel()
    mean = image_data.mean()
    std = image_data.std()     
    
    return tmp, filtered_idx, mean, std, mask_gm, orig_shape

def load_and_preprocess_data(args, modality_dict):
    # Paths
    image_path = f'{args.data_dir}/adni/img/h5py_files'
    data_path = f'{args.data_dir}/adni/meta'
    label_df = pd.read_csv(os.path.join(data_path,'longit_m061224_DX.csv'), index_col='PTID')

    # Keep Month 0 only
    label_df = label_df[label_df['Month'] == 0.0]
    # Remove rows where both mri and fdg are missing
    label_df = label_df.dropna(subset=['imgid_mri','imgid_fdg'], how='all')

    label_df = label_df[~label_df.index.duplicated(keep='first')]

    # If unimodal, remove rows where that modality is missing
    if args.modality == 'M':
        label_df = label_df.dropna(subset=['imgid_mri'])
    elif args.modality == 'F':
        label_df = label_df.dropna(subset=['imgid_fdg'])

    label_df['DX'] = label_df['DX'].map({'CN':0, 'MCI':1, 'Dementia':2})
    labels = label_df['DX'].values.astype(np.int64)
    n_labels = len(set(labels))

    data_split = pd.read_csv(f'{args.data_dir}/adni/meta/data_splits.csv')
    train_ids = list(data_split[data_split['split0'] == 'train']['PTID'].drop_duplicates())
    valid_ids = list(data_split[data_split['split0'] == 'val']['PTID'].drop_duplicates())
    test_ids = list(data_split[data_split['split0'] == 'test']['PTID'].drop_duplicates())

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    train_transforms = {}
    eval_transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0],2), dtype=bool) # MF order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    orig_img_shapes = {}
    # Load modalities
    if 'M' in args.modality or 'm' in args.modality:
        mod = 'mri'
        arr, filtered_idx, mean, std, mask, orig_shape = load_and_preprocess_image_data(mod, args.data_dir, image_path, label_df, id_to_idx)
        observed_idx_arr[:, modality_dict['mri']] = arr[:, 0] != -2
        for idx in filtered_idx:
            update_modality_combinations(idx, 'M')

        data_dict['mri'] = np.array(arr)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['mri'] = torch.nn.Sequential(
            DenseNet(spatial_dims=3, in_channels=1, out_channels=args.hidden_dim, dropout_prob=0.3, init_features=64, growth_rate=64, num_patches=args.num_patches).to(args.device),
            )
        encoder_dict['mri2fdg'] = torch.nn.Sequential(
            DenseNet(spatial_dims=3, in_channels=1, out_channels=args.hidden_dim, dropout_prob=0.3, init_features=64, growth_rate=64, num_patches=args.num_patches).to(args.device),
            )

        input_dims['mri'] = arr.shape[1]

        # Defining transformations
        norm_range, min_p_m, max_p_m = DefineRangeNormalization()
        train_transforms['mri'] = Compose([
                                    ToTensor3D(),
                                    norm_range,
                                    Normalize(mean=[mean], std=[std]),
                                    # random translation
                                    tio.transforms.RandomAffine(scales=(1,1), degrees = (0,0,0,0,0,0), translation =(-10,10,-5,5,-10,10),isotropic  = True, default_pad_value  = 0, p=0.5),
                                    # random rotation
                                    tio.transforms.RandomAffine(scales=(1,1), degrees = (-10,10,-5,5,-5,5), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
                                    # random zoom
                                    tio.transforms.RandomAffine(scales=(0.9,1.1), degrees = (0,0,0,0,0,0), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
                                ])
        eval_transforms['mri'] = Compose([
                                    ToTensor3D(),
                                    norm_range,
                                    Normalize(mean=[mean], std=[std]),
                                ])
        orig_img_shapes['mri'] = orig_shape
        masks['mri'] = mask

    if 'F' in args.modality or 'f' in args.modality:
        mod = 'fdg'
        arr, filtered_idx, mean, std, mask, orig_shape = load_and_preprocess_image_data(mod, args.data_dir, image_path, label_df, id_to_idx)
        observed_idx_arr[:, modality_dict['fdg']] = arr[:, 0] != -2
        for idx in filtered_idx:
            update_modality_combinations(idx, 'F')

        data_dict['fdg'] = np.array(arr)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['fdg'] = torch.nn.Sequential(
            DenseNet(spatial_dims=3, in_channels=1, out_channels=args.hidden_dim, dropout_prob=0.3, init_features=64, growth_rate=64, num_patches=args.num_patches).to(args.device),
            )
        encoder_dict['fdg2mri'] = torch.nn.Sequential(
            DenseNet(spatial_dims=3, in_channels=1, out_channels=args.hidden_dim, dropout_prob=0.3, init_features=64, growth_rate=64, num_patches=args.num_patches).to(args.device),
            )

        input_dims['fdg'] = arr.shape[1]

        # Defining transformations
        norm_range, min_p_m, max_p_m = DefineRangeNormalization()
        train_transforms['fdg'] = Compose([
                                    ToTensor3D(),
                                    norm_range,
                                    Normalize(mean=[mean], std=[std]),
                                    # random translation
                                    tio.transforms.RandomAffine(scales=(1,1), degrees = (0,0,0,0,0,0), translation =(-10,10,-5,5,-10,10),isotropic  = True, default_pad_value  = 0, p=0.5),
                                    # random rotation
                                    tio.transforms.RandomAffine(scales=(1,1), degrees = (-10,10,-5,5,-5,5), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
                                    # random zoom
                                    tio.transforms.RandomAffine(scales=(0.9,1.1), degrees = (0,0,0,0,0,0), translation =(0,0,0,0,0,0),isotropic  = True, default_pad_value  = 0, p=0.5),
                                ])
        eval_transforms['fdg'] = Compose([
                                    ToTensor3D(),
                                    norm_range,
                                    Normalize(mean=[mean], std=[std]),
                                ])
        masks['fdg'] = mask
        orig_img_shapes['fdg'] = orig_shape


    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    common_idxs = set.intersection(*common_idx_list)
    common_test_idxs = list(common_idxs & set(test_idxs))

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, common_test_idxs, n_labels, input_dims, train_transforms, eval_transforms, masks, observed_idx_arr, full_modality_index, orig_img_shapes


def collate_fn(batch):
    data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    if labels[0].dtype == np.float64:
        labels = torch.tensor(labels, dtype=torch.float)
    else:
        labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds

def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, common_test_ids, batch_size, num_workers, pin_memory, input_dims, train_transforms, eval_transforms,  masks, orig_img_shapes, use_common_ids=True):
    if 'mri' in list(data_dict.keys()):
        train_transfrom = train_transforms['mri']
        val_transform = test_transform = eval_transforms['mri']
        # val_transform = test_transform = False
        mask = masks['mri']
    if 'fdg' in list(data_dict.keys()):
        train_transfrom = train_transforms['fdg']
        val_transform = test_transform = eval_transforms['fdg']
        # val_transform = test_transform = False
        mask = masks['fdg']
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, orig_img_shapes, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, orig_img_shapes, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, orig_img_shapes, use_common_ids)
    common_test_dataset = MultiModalDataset(data_dict, observed_idx, common_test_ids, labels, input_dims, test_transform, mask, orig_img_shapes, use_common_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    common_test_loader = DataLoader(common_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader, common_test_loader

# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index
