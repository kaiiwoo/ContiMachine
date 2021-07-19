# -*- coding: utf-8 -*- 
# from StoryGAN_pororo.miscc.utils import KL_loss
from genericpath import exists
from numpy.lib.arraysetops import isin
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image, make_grid
import numpy as np
import os
from model import STORYGAN, StoryEncoder, TextEncoder,  D_IMG, D_STY #Story_D, Image_D,
import time
from copy import deepcopy
from tqdm import tqdm
# from tensorboardX import SummaryWriter 

class Researcher(object):
    def __init__(self, args, image_loader, story_loader):
        self.args = args
        self.image_loader = image_loader
        self.story_loader = story_loader
            
        self.imagedata = None
        self.testdata = None
        self.logger = None

    def sample_real_image(self):
        if self.imagedata is None:
            self.imagedata = enumerate(self.image_loader)
        batch_idx, minibatch = next(self.imagedata)
        b = minibatch
        for k, v in b.items():
            if (k !='text') and (torch.cuda.is_available()):
                b[k] = v.cuda()
                
        if batch_idx == len(self.image_loader) -1:
            self.imagedata = enumerate(self.image_loader)
        return b


    def images_to_numpy(self, tensor):
        generated = tensor.data.numpy().transpose(1,2,0) #무조건 H,w,1로.
        lower = generated[generated < -1]
        upper = generated[generated > 1]
        lower = -1
        upper = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype('uint8')

    
    def save_model(self, D1, D2, G, epoch):
        torch.save(D1.state_dict(), '{}/D1_{}.pth'.format(self.model_dir, epoch))
        torch.save(D2.state_dict(), '{}/D2_{}.pth'.format(self.model_dir, epoch))
        torch.save(G.state_dict(), '{}/G_{}.pth'.format(self.model_dir, epoch))
        
        
    # B,L,3,64,64
    # 24, 4, 3, 64, 64
    def save_stories(self, r_story, f_story, epoch, video_len=5, is_test=False):
        B = r_story.shape[0]
        r_story = r_story.cpu()
        f_story = f_story.cpu()

        all_f = []
        all_r = []
        for i in range(B):
            all_f.append(make_grid(f_story[i], video_len))
            all_r.append(make_grid(r_story[i], video_len))
        f_images, r_images = make_grid(all_f, 1), make_grid(all_r, 1)
        f_images, r_images = self.images_to_numpy(f_images), self.images_to_numpy(r_images)
        all_images = np.concatenate([f_images, r_images], axis=1)
        all_images = Image.fromarray(all_images)
        
        if not is_test:
            all_images.save('{}/fake_samples_{}.png'.format(self.image_dir, epoch))
        else:
            all_images.save('{}/test_samples_{}.png'.format(self.image_dir, epoch))
            
            
    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
  
    # Kl Divergence Regularization Term 계산.
    def KL_loss(self, mu, logvar):
        single_kl = 1 + logvar -(mu.pow(2) + torch.exp(logvar)) 
        kl_loss = -0.5 * torch.mean(single_kl)
        return self.args.kl_coeff *kl_loss

    # 판별자 로스 계산
    def compute_d_loss(self, real_sample, fake_sample, global_info, r_label,  f_label, D, is_image=False, do_r1=True):
        """
        2.뽀로로
        (image)
        - real_sample: B, C, H, W
        - fake_sample: B, C, H, W
        - image_global_info : B, 'tD + 9' + hD  = B, 256 + 9 + 128 = B, 393


        (story)
        - real_sample: B, C, L, H, W 
        - fake_sample: B, L, C, H, W
        global_s_info- info_story : B, hD + tD + 9 = B, 393
        """

        criterion = nn.BCELoss()
        fake_sample = fake_sample.detach()
        global_info = global_info.detach().requires_grad_(True)
        
        # feature representation extraction in Discriminator
        real_represent = nn.parallel.data_parallel(D, (real_sample), [0])
        fake_represent = nn.parallel.data_parallel(D, (fake_sample), [0])
        
        # clevr 학습시에만 필요한 분기
        if not is_image:
            global_info  = global_info.mean(1)
        
        # 로짓계산 
        inputs = (real_represent, global_info) 
        real_logits = nn.parallel.data_parallel(D.get_cond_logits, inputs, [0])
        errD_real = criterion(real_logits, r_label)
        
        if do_r1:
            loss_Dr1 = 0
            r1_gamma = 10
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_represent, global_info], create_graph=True, only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1,2,3])
            loss_Dr1 = r1_penalty * (r1_gamma / 2)
            errD_real = (errD_real + loss_Dr1).mean()
        
        inputs = (fake_represent, global_info) 
        fake_logits = nn.parallel.data_parallel(D.get_cond_logits, inputs, [0])
        errD_fake = criterion(fake_logits, f_label)
        
        # wrong pairs
        inputs = (real_represent[:(self.args.batch_size-1)], global_info[1:])
        wrong_logits = \
            nn.parallel.data_parallel(D.get_cond_logits, inputs, [0])
        errD_wrong = criterion(wrong_logits, f_label[1:])
        errD = errD_real + (errD_fake + errD_wrong) * 0.5

        return self.args.alpha * errD
        
    # 생성자 loss 계산. 
    def compute_g_loss(self, f_story, global_info, r_label, D, is_image=False):
        criterion = nn.BCELoss()
        global_info = global_info.detach()
        
        fake_features = nn.parallel.data_parallel(D, f_story, [0])    #이미지 - 24, 768, 4, 4    스토리 - 24, 3072, 4, 4

        # clevr 학습시에만 필요.
        if not is_image:
            global_info = global_info.mean(1)
        
        inputs = (fake_features, global_info)
        fake_logits = nn.parallel.data_parallel(D.get_cond_logits, inputs, [0])
        errD_fake = criterion(fake_logits, r_label)
        return  errD_fake

    
    def init_networks(self, dataset):
        assert isinstance(dataset, str)
        # clevr args.hid_dim = 96
        if dataset=='clevr':
            net_G = STORYGAN().cuda()
            net_G.apply(self.weight_init)
            
            image_D = D_IMG(self.args.hid_dim).cuda()  #hid_dim = 96
            image_D.apply(self.weight_init)
            
            story_D = D_STY(self.args.hid_dim).cuda()
            story_D.apply(self.weight_init)
            print('initialized networks for training Clevr dataset')
        
        # pororo args.hid_dim = 
        elif dataset=='pororo':
            net_G = STORYGAN(text_dim=256, seq_len=5, dataset=dataset).cuda()
            net_G.apply(self.weight_init)
            
            # 96 --> 
            image_D = D_IMG(124, text_dim=256, dataset=dataset).cuda() 
            image_D.apply(self.weight_init)
            
            story_D = D_STY(124, text_dim=256, dataset=dataset).cuda()
            story_D.apply(self.weight_init)
            print('initialized networks for training Pororo dataset')
            
        else:
            raise NotImplementedError

        return net_G, image_D, story_D

    def train(self): 
        output_dir = os.path.join(self.args.output_dir, self.args.dataset)
        self.output_dir = os.path.join(output_dir, 'model')
        self.image_dir = os.path.join(output_dir, 'image')
        self.log_dir = os.path.join(output_dir, 'log')

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.image_dir)
            os.makedirs(self.log_dir)

        # self.logger = SummaryWriter(self.log_dir)

        # 데이터셋에 맞는 네트워크 초기화
        net_G, image_D, story_D = self.init_networks(self.args.dataset)
        

        netG_para = []
        for p in net_G.parameters():
            if p.requires_grad:
                netG_para.append(p)
        net_G_optim = torch.optim.Adam(netG_para, lr=self.args.G_lr,
                                betas=(0.5, 0.999))
        
        story_D_optim = torch.optim.Adam(story_D.parameters(), lr=self.args.D_lr, betas= (0.5, 0.999))
        image_D_optim = torch.optim.Adam(image_D.parameters(), lr=self.args.D_lr, betas= (0.5, 0.999))
        
        """
        1. Clevr
        Image Data: {'image':image,                 #단일 이미지                    #torch.size([3,64,64])
                    'description':description,      #해당 씬의 정보. 물체 수 만큼 행(dim=18)들 채워있음.          #(75,)
                    'label':super_label,            #해당 씬에 등장한 모든 물체의 attribute들은 1로 돼있는 원핫 attribute vector (shape, color, material, size)               #(15,)
                    'contents':contents}            #해당 스토리내 이미지들의 descriptions. 물체 등장순서대로 존재. 따라서 첫 des는 첫행만, 두 번째 des는 두행까지, 3번째는 3행까지.. 채워있음.  #(4, 75)
                    
        imagedata['contents'] == storydata['descriptions']
                            
        Story Data:  {'images':images,              #이미지시퀀스                   #torch.size([3,4,64,64])
                    'descriptions':descriptions,    #해당 스토리내 이미지들의 descriptions. 물체 등장순서대로 존재. 따라서 첫 des는 첫행만, 두 번째 des는 두행까지, 3번째는 3행까지.. 채워있음.                      #(4, 75)
                    'labels':super_label}           #각 행은 해당 씬에서 등장하는 모든 물체들의 attribute는 1로 표현되어있는 attribute vector. 따라서 아래 행일수록 더 많이 1로 채워져있음. (shape, color, material, size)      #(4, 15)


        2. Pororo
        -StoryDataset
        return {'images': image, 'text':text, 'description': des, 
                'subtitle': subs, 'images_numpy':image_numpy, 'labels':labels}

        -imageDataset
        images : [B,C,S,H,W]
        return {'images': image, 'text':text, 'description': des,  
            'subtitle': subs, 'labels':label, 'content': content}
            
        """
        
        start_time = time.time()
        print("Start Training...")
        for epoch in tqdm(range(self.args.max_epoch)): 
            # learing rate decay
            if (epoch + 1) % self.args.lr_decay_epoch == 0: 
                self.args.D_lr = self.args.D_lr * 0.5
                for param in story_D_optim.param_groups: 
                    param['lr'] = self.args.D_lr
                for param in image_D_optim.param_groups:
                    param['lr'] = self.args.D_lr
                    
                self.args.G_lr =  self.args.G_lr*0.5
                for param in net_G_optim.param_groups:
                    param['lr'] = self.args.G_lr
            
            for idx, story_batch in enumerate(self.story_loader, 0):
                
                real_label = torch.ones(story_batch['images'].shape[0], requires_grad=False)
                fake_label = torch.zeros(story_batch['images'].shape[0], requires_grad=False)
                if torch.cuda.is_available():
                    real_label = real_label.cuda()
                    fake_label = fake_label.cuda()
                
                r_img_batch = self.sample_real_image()
                
                if self.args.dataset == 'pororo':
                    r_img = r_img_batch['images']                                  # B,C,H,W
                    r_img_description = r_img_batch['description'][:, :256]        # B,D
                    r_img_label = r_img_batch['labels']                            # B,9 (9 무엇?)
                    r_img_contents = r_img_batch['content'][:, :, :256]            # B,S,D : 24 5 256       #원래 D크기 : 365
                    
                    r_story = story_batch['images']                                # B,C,S,H,W : 24 3 5 64 64
                    r_story_contents = story_batch['description'][:, :, :256]      # B,S,D          # 원래 D 크기 :356
                    r_story_labels = story_batch['labels'][:, :, :256]             # B,S,9          # 사실 labels는 슬라이싱 없어도 됨. 어차피 크기 9니까..
                    r_texts = story_batch['text']                                  # len = 5 

                    # 7/15 나중에 디버깅 하자..
                    # if idx==0:
                    #     torchvision.utils.save_image(r_story[0].type(np.uint8), '.first.png')
                    #     pickle.save(r_texts[0], '.first.txt') 
                else:
                    r_img = r_img_batch['images']                    # B,3,H,W
                    r_img_description = r_img_batch['description']   # B,75
                    r_img_label = r_img_batch['label']               # B,15
                    r_img_contents = r_img_batch['contents']         # B,S,75
                    
                    r_story = story_batch['images']                  # B,S,3,H,W 
                    r_story_contents = story_batch['descriptions']   # B,S,75 
                    r_story_labels = story_batch['labels']           # B,S,15 씬에 등장하는 모든 물체 속성 누적합


                if torch.cuda.is_available():
                    r_img = r_img.cuda()
                    r_img_description = r_img_description.cuda()
                    r_img_label = r_img_label.cuda()
                    r_img_contents = r_img_contents.cuda()
                    
                    r_story = r_story.cuda()
                    r_story_contents = r_story_contents.cuda()
                    r_story_labels = r_story_labels.cuda()
                    
                # 뽀로로인경우 descriptions들 label과 히든디멘션 concat. 
                if self.args.dataset == 'pororo':
                    r_img_description = torch.cat((r_img_description, r_img_label), 1) #B, D + 9
                    r_story_description = torch.cat((r_story_contents, r_story_labels), 2) #B, S, D + 9
                else:
                    r_story_description = r_story_contents

                # generate image & stories
                fake_img, img_mu, _ = net_G.make_img(r_img_description, r_img_contents)            #B, C, H, W
                fake_story2, story_mu, _ = net_G.make_stories(r_story_description, r_story_contents)    #B, L, C, H, W


                story_D.zero_grad()
                image_D.zero_grad()

                # GLOBAL INFO FOR COMPUTING LOSSES 
                if self.args.dataset == 'pororo':
                    character_mu = (r_story_labels.mean(1)>0).type(torch.FloatTensor).cuda() #B, S, 9 -> B, 9
                    global_i_info = torch.cat((r_img_description, img_mu), 1) #B, 'tD + 9' + hD 
                    global_s_info = torch.cat((story_mu, r_story_contents.mean(1).squeeze(), character_mu), 1) #B, hD + tD + 9
                else:
                    global_i_info = r_img_description
                    global_s_info = r_story_contents


                # 1. 이미지에 대한 로스 계산
                image_d_loss = self.compute_d_loss(r_img,
                                            fake_img,       # 각 이미지에 대해서 판단
                                            global_i_info,  # global condition
                                            real_label,
                                            fake_label, 
                                            image_D, is_image=True, do_r1=True)
                image_d_loss  = self.args.alpha * image_d_loss 
                
                # 2. 스토리에 대한 로스 계산
                # 뽀로로는 is_image True 
                if self.args.dataset == 'pororo':
                    story_d_loss = self.compute_d_loss(r_story,
                                                fake_story2, 
                                                global_s_info,  # global condition
                                                real_label,
                                                fake_label, 
                                                story_D, True, do_r1=True) 
                    story_d_loss =  self.args.beta * story_d_loss 

                else: 
                    story_d_loss = self.compute_d_loss(r_story,
                                                fake_story2, 
                                                global_s_info,  # global condition
                                                real_label,
                                                fake_label, 
                                                story_D, do_r1=True)
                    story_d_loss =  self.args.beta * story_d_loss 
                
                step = epoch * len(self.story_loader) + idx
                if self.logger:
                    self.logger.add_scalar('D_st_loss', story_d_loss, step)
                    self.logger.add_scalar('D_img_loss', image_d_loss, step)
                    self.logger.add_scalar('D_total', story_d_loss + image_d_loss, step)

                image_d_loss.backward()
                story_d_loss.backward()
                image_D_optim.step()
                story_D_optim.step()
                
                
                # 생성자 업데이트
                for i in range(2):
                    net_G.zero_grad()
                    
                    fake_img, mu, _ = net_G.make_img(r_img_description, r_img_contents)           
                    fake_story2, mu2, _ = net_G.make_stories(r_story_description, r_story_contents) 

                    g_loss1 = self.compute_g_loss(fake_img,
                                                global_i_info,
                                                real_label,
                                                image_D, is_image=True)

                    g_loss2 = self.compute_g_loss(fake_story2,
                                                global_s_info,
                                                real_label,
                                                story_D, True)
                    
                    g_loss = g_loss1 + g_loss2 

                    if self.logger:
                        self.logger.add_scalar('G_st_loss', g_loss2, step)
                        self.logger.add_scalar('G_img_loss', g_loss1, step)
                        self.logger.add_scalar('G_total', g_loss, step)

                    g_loss.backward()
                    net_G_optim.step()
                    
                print('\n')
                print('Epoch:{}\tIter:[{}/{}]\tstoryD_loss:{}\timageD_loss:{}\tG_loss:{}'.format(epoch+1, idx+1, len(self.story_loader), story_d_loss, image_d_loss, g_loss))
                            
            with torch.no_grad():
                print("saving sample stories and models...")
                sample_story,_,_  = net_G.make_stories(r_story_contents, r_story_contents)
            self.save_stories(r_story, sample_story, epoch+1)
            self.save_model(image_D, story_D, net_G, epoch)
                            
        end_time = time.time()
        print("----------Training has Finished-----------------")
        print("Total Consumed Trainig Time:{}min".format((end_time - start_time) // 60))
        
        
