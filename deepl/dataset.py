import torch
import numpy as np
import torch.utils.data as DataLoader,Dataset
class Dataset_name(Dataset):
    def __init__(self,flag='train'):
        assert 'flag' in ['train','val'.'test']
        self.flag = flag
        self.__load_data__()
    def __getitem__(self,index):
        pass
    def __len__(self):
        pass
    def __load_data__(self):
        pass
        print(
            "train_X.shape:{}\ntrain_Y.shape:{}\nvalid_X.shape:{}\nvalid_Y.shape:{}\n"
            .format(self.train_X.shape, self.train_Y.shape, self.valid_X.shape, self.valid_Y.shape))
train_Dataset = Dataset_name(flag='train')
train_dataloader = DataLoader(dataset=train_Dataset,batch_size=64,shuffle = True)
valid_dataset = Dataset_name(flag='val')
valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=64,shuffle = True)