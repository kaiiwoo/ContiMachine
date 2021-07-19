# -*- coding: utf-8 -*- 

import torch
from PIL import Image
import numpy as np
import os

# Super Label vs Label? 


# 단일 이미지 데이터
class ImageData(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_transform=None, seq_len=4, is_train=True):
        super().__init__()
        self.image_transform = image_transform
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.image_dir = os.path.abspath(os.path.join(data_dir, 'images')) #StoryGAN/clevr_dataset/images
        self.attribute_file = os.path.abspath(os.path.join(data_dir, 'CLEVR_dict.npy'))
        
        self.npy_dict = np.load(self.attribute_file, allow_pickle=True, encoding='latin1').item() # filename : attribute
        
        if is_train:
            self.start_id=100
            self.end_id=172
        else:
            self.start_id=172
            self.end_id=196
            
    def __getitem__(self, idx):
        idx = self.start_id + idx
        scene_idx = np.random.randint(1, self.seq_len +1)
        scene_file = 'CLEVR_new_{:06}_{}.png'.format(idx, scene_idx)
        # 단일이미지
        image = Image.open(os.path.join(self.image_dir, scene_file))
        image = np.array(image)
        if self.image_transform:
            image = self.image_transform(image)
        
        # description = 해당이미지 파일의 feature vectors
        description = self.npy_dict[scene_file].astype(np.float32)
        super_label = description[:15]
        contents = []
        
        # 왜 descritipon은 video length만큼 추출?
        for j in range(1, self.seq_len+1):
            iter_file_id = 'CLEVR_new_{:06}_{}.png'.format(idx, j)
            des = self.npy_dict[iter_file_id].astype(np.float32) #같은 id를 가진(같은 묶음인) 이미지들의 description context에 추가.
            contents.append(np.expand_dims(des, axis=0))
            
        for i in range(1, self.seq_len): # 1,2,3 일때만. 도대체왜?
            super_label = super_label + description[i*18 : i*18 + 15]
            
        super_label = super_label.reshape(-1)
        larger_than_1 = super_label[super_label>1] 
        larger_than_1 = 1
        contents = np.concatenate(contents, axis=0)
        return {'image':image,
                'description':description, #해당 씬의 description
                'label':super_label,
                'contents':contents}    # 같은 에피소드 내 다른 이미지들의 descriptions. 이 무슨의미?
        
    def __len__(self):
        return self.end_id - self.start_id + 1
        


class StoryData(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_transform=None, seq_len=4, is_train=True):
        super().__init__()
        self.image_transform = image_transform
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.image_dir = os.path.abspath(os.path.join(data_dir, 'images')) #StoryGAN/clevr_dataset/images
        self.attribute_file = os.path.abspath(os.path.join(data_dir, 'CLEVR_dict.npy')) #StoryGAN/clevr_dataset/
        
        self.npy_dict = np.load(self.attribute_file, allow_pickle=True, encoding='latin1').item() # filename : attribute
        
        if is_train:
            self.start_id=100
            self.end_id=172
        else:
            self.start_id=172
            self.end_id=196

    
    def __getitem__(self, idx):
        idx = self.start_id + idx
        scene_idx = np.random.randint(1, self.seq_len +1)
        scene_file = 'CLEVR_new_{:06}_{}.png'.format(idx, scene_idx)
        
        description = self.npy_dict[scene_file].astype(np.float32)
        
        super_label = description[:15]
        
        images = []
        descriptions = []
        super_labels = []
        
        video = []
        for j in range(self.seq_len):
            iter_file_id = 'CLEVR_new_{:06}_{}.png'.format(idx, j+1)
            # print(iter_file_id)
            image = Image.open(os.path.join(self.image_dir, iter_file_id))
            image = np.array(image)
            images.append(np.expand_dims(image, 0)) # 1CHW?
            
            des = self.npy_dict[iter_file_id].astype(np.float32)
            descriptions.append(np.expand_dims(des.astype(np.float32), axis=0)) 
            
            label_origin = descriptions[-1].reshape(-1)
            
            # labels.append(label_origin[j*18 + 3: j*18 + 11])
            super_labels.append(label_origin[j*18 : j*18 + 15])


        # label[0] = np.expand_dims(label[0], axis = 0)
        super_labels[0] = np.expand_dims(super_labels[0], axis=0)
        for i in range(1,4):
            # label[i] += label[i-1]
            super_labels[i] = np.expand_dims(super_labels[i], axis=0)
            super_labels[i] = super_labels[i] + super_labels[i-1]
            # temp = label[i].reshape(-1) #왜 temp를 썼지? 바로 label, super_labels에 적용하면 안되나?
            
            super_temp = super_labels[i].reshape(-1)
            # temp[temp>1] = 1
            
            larger_than_1 = super_temp[super_temp>1] 
            larger_than_1 = 1
            # label[i] = np.expand_dims(temp, axis = 0)
            super_labels[i] = np.expand_dims(super_temp, axis = 0)
            
        descriptions = np.concatenate(descriptions, axis=0)
        # labels = np.concatenate(labels, axis=0) 
        super_labels = np.concatenate(super_labels, axis=0)
        images = np.concatenate(images, axis=0) # transform 이전 images shape : 4, 240, 320, 4
        
        if self.image_transform:
            for img in images:
                video.append(self.image_transform(img)) 
        video = torch.stack(video)

        return {'images':video,
                'descriptions':descriptions,
                'labels':super_labels}
        
    def __len__(self):
        return self.end_id - self.start_id + 1


def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return loader
 