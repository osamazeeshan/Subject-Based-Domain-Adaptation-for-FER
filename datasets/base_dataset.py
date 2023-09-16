import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from datasets.dataset import PainDatasets

SPLIT = 0.90

class BaseDataset():
    """
    Dataset base class: all the other dataloader classes will be inherit from base class 
    """
    def load_tar_acpl_data(target_dataset, batch_size, split = False):
        _, val_set = torch.utils.data.random_split(target_dataset, [len(target_dataset) - int((1-SPLIT)*len(target_dataset)), int((1-SPLIT)*len(target_dataset))]) if split else (target_dataset, None)
        data_loader_train = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        data_loader_val = DataLoader(val_set, batch_size=1, shuffle=True, drop_last=False) if split else None

        return data_loader_train, data_loader_val

    def load_pain_dataset(dataset_path, label_path_train, label_path_val, batch_size, phase):
        try: 
            val_idx = []
            train_idx = []
            shuffle = False
            test_loader = None

            transform_dict = {
            'src': transforms.Compose(
            [
                transforms.Resize((100,100)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
            ]),
            'tar': transforms.Compose(
            [
                transforms.Resize((100,100)),
            ])}

            dataset_train = PainDatasets(dataset_path, label_path_train, transform=transform_dict[phase])
            
            """
                condition to split train and val if no validation is defined
            """
            if label_path_val is not None:
                dataset_val = PainDatasets(dataset_path, label_path_val)
                shuffle = True
            else:
                shuffled_indices = np.random.permutation(len(dataset_train))
                if phase == 'tar':
                    train_idx = shuffled_indices[:int(0.70*len(dataset_train))]
                    val_idx = shuffled_indices[int(0.70*len(dataset_train)):int(0.80*len(dataset_train))]
                    test_idx = shuffled_indices[int(0.80*len(dataset_train)):int(1.0*len(dataset_train))]
                else:
                    train_idx = shuffled_indices[:int(0.80*len(dataset_train))]
                    val_idx = shuffled_indices[int(0.80*len(dataset_train)):int(1.0*len(dataset_train))]

                dataset_val = dataset_train

            train_loader = DataLoader(  dataset_train,
                                        batch_size=batch_size, 
                                        drop_last=True,
                                        num_workers=1, 
                                        pin_memory=True,
                                        shuffle=shuffle,
                                        sampler=SubsetRandomSampler(train_idx) if len(train_idx) != 0 else None
                                    )

            val_loader = DataLoader(    dataset_val, 
                                        batch_size=1, 
                                        drop_last=False, 
                                        num_workers=1, 
                                        pin_memory=True,
                                        sampler=SubsetRandomSampler(val_idx) if len(val_idx) != 0 else None
                                    )
            if phase == 'tar':
                test_loader = DataLoader(   dataset_train, 
                                            batch_size=1, 
                                            drop_last=False, 
                                            num_workers=1, 
                                            pin_memory=True,
                                            sampler=SubsetRandomSampler(test_idx) if len(test_idx) != 0 else None
                                        )                        

            return train_loader, val_loader, test_loader

        except Exception as e:
            print("Error: ", e)        