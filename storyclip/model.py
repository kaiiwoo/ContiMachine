# -*- coding: utf-8 -*- 
from unicodedata import bidirectional
import numpy as np 
from dfn import DynamicFilterLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.utils import spectral_norm
if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


def conv3x3(in_planes, out_planes, stride=1, use_spectral_norm=True):
    "3x3 convolution with padding"
    if use_spectral_norm:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block
    
class StoryEncoder(nn.Module):
    def __init__(self, text_dim=75, out_dim=128, seq_len=4, bias=True, lr_mul=1, dataset='clevr'):
        super().__init__()
        self.out_dim = out_dim
        self.fc = nn.Linear(text_dim * seq_len, 2*out_dim, bias=bias)
        self.dataset = dataset

    def encode(self, text_emb):
        out = F.relu(self.fc(text_emb))
        mu = out[:, :self.out_dim]
        logvar = out[:, self.out_dim:]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """
        x = full_story (ex) r_img_contents : B, L, D
        clevr : B, L, 75
        Pororo : B, L, 256
        """
        x = x.reshape(x.shape[0], -1) #B,LD
        mu, logvar = self.encode(x)
        z_code = self.reparameterize(mu, logvar)
        return z_code, mu, logvar #B128
    

# Context Encoder
class TextEncoder(nn.Module):
    def __init__(self, text_dim=75, hid_dim=128, seq_len=4, noise_dim=10, dataset='clevr'):
        super().__init__()

        if dataset=='clevr':
            self.linear = nn.Sequential(
            nn.Linear(text_dim + noise_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
            )
        elif dataset=='pororo':
            self.linear = nn.Sequential(
            nn.Linear(text_dim + 9 + noise_dim, hid_dim), #+= label_num
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
            )
            
        self.noise_dim = noise_dim
        self.video_len = seq_len 
        self.gru_bottom = nn.GRU(hid_dim, hid_dim, batch_first=True,bidirectional=True )
        # self.text2gist = nn.GRU(hid_dim, hid_dim, batch_first=True,)
        self.dataset = dataset

    def inject_noise(self, text):
        if text.ndim==3:
            B, L, _ = text.shape
            noise = torch.randn(B, L, self.noise_dim).cuda()
        else:
            B, _ = text.shape
            noise = torch.randn(B, self.noise_dim).cuda()
        concat_list = [text, noise]
        out = torch.cat(concat_list, -1)
        return out
    
    def forward(self, x, hid_init):
        """
        x = input_desc 
        (ex) r_img_description : B, L, tD or B, L, tD+9 

        clevr : B, L, tD
        Pororo : B, L, tD + 9
        """
        
        B, _, _ = x.shape
        x = self.inject_noise(x) #B,L, tD + nD 
        x = x.view(-1, x.shape[-1]) #B*L, tD+nD
        x = self.linear(x) #B*L, hD
        D = x.shape[-1]
        x = x.view(B, -1, D) #B, L, hD
        
        output1, g_n = self.gru_bottom(x)
        
        return (output1, g_n)
        

# Generator(self.args.n_channels * 2 = 128)
class Generator(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        self.downsamples = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, 2, 1, bias=False), #1 -> 1536
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_channels, n_channels//2, 4, 2, 1, bias=False), #1536
            nn.BatchNorm2d(n_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.upsamples=nn.Sequential(
            #1536 -> 512
            #4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest', ), # == mode=nearest
            nn.Conv2d(n_channels, n_channels//2, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(n_channels//2),
            nn.LeakyReLU(),
            
            #512 -> 256
            #8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest', ),
            nn.Conv2d(n_channels//2, n_channels//4, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(n_channels//4),
            nn.LeakyReLU(),
            
            #256 -> 128
            #16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest', ),
            nn.Conv2d(n_channels//4, n_channels//8, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(n_channels//8),
            nn.LeakyReLU(),
            
            #128 -> 64
            #32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest', ),
            nn.Conv2d(n_channels//8, n_channels//16, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(n_channels//16),
            nn.LeakyReLU(),
            
            #64 -> 3(RGB)
            #64x64
            nn.Conv2d(n_channels//16, 3, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
    
    def forward(self, x):
        shrinked = self.downsamples(x)
        upsample_input = torch.cat([shrinked, shrinked], dim=1)
        return self.upsamples(upsample_input)
    

################################################################################################################

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, video_len = 1, cond_text=True, dataset='clevr'):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf # arg.hid_dim  현재 124
        self.ef_dim = nef # Clevr : text_dim / Pororo : 128 + text_dim + 9
        self.cond_text = cond_text
        self.video_len = video_len
        self.dataset =  dataset
        
        if cond_text:
           #이미지 인코더(Image_D의 layer 5* & 6)
            self.conv1 = nn.Sequential(
                conv3x3(ndf * 8 * video_len, ndf * 8) if dataset == 'clevr' else  conv3x3(ndf * 8 + nef, ndf * 8), 
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
            
            # 이미지 인코더 final sigmoid layer  
            self.conv2 = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
                )
            
            # 텍스트 인코더
            self.convc = nn.Sequential(
                conv3x3(self.ef_dim, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
            
        # 텍스트 컨디션 안받고 순수 이미지에 대해서만 판별할 때. 
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
                )

    def forward(self, h_code, c_code=None):
        """
        0.공통
        - c_code: B, ef_dim, 4, 4 매핑

        1. clevr
        - c_code를 텍스트 인코더(self.convc)에 태워줌
        - sigmoid layer엔 self.convc(c_code) * self.conv1(h_code)가 입력으로

        h_code(이미지임베딩) = 이미지 : 24, 768, 4, 4  스토리 : 24, 3072, 4, 4
        c_code(텍스트임베딩) = 이미지 : 24, 75 (im_motion_input)
                              스토리 : 24, 75 (st_content_input.mean(1)) 

        2. 뽀로로
        - 텍스트 인코더 사용안함.
        - c_code와 h_code와 concat 후 self.conv1 --> sigmoid layer 한번에 feed forward.

        h_code : B, df_dim(124)*8, H, W
        c_code(image/story_global_info) : B, hD + tD + 9 = B, 393

        """
        if self.cond_text and c_code is not None:
            if self.dataset == 'pororo':
                c_code = c_code.view(-1, self.ef_dim, 1, 1)
                c_code = c_code.repeat(1, 1, 4, 4) 
                h_c_code = torch.cat((h_code, c_code), 1)
                h_c_code = self.conv1(h_c_code)
            else:
                c_code = c_code.view(-1, self.ef_dim, 1, 1)
                c_code = c_code.repeat(1, 1, 4, 4) 
                c_code = self.convc(c_code)
                h_code = self.conv1(h_code) 
                h_c_code = h_code * c_code # 두 피쳐의 원소곱
        else:
            h_c_code = h_code

        output = self.conv2(h_c_code)
        return output.view(-1)

class D_IMG(nn.Module):
    def __init__(self, hid_dim, text_dim=75, use_categories = True, dataset='clevr'):
        super(D_IMG, self).__init__()
        self.hid_dim = hid_dim 
        self.text_dim = text_dim
        self.dataset = dataset
        self.define_module(use_categories)

    def define_module(self, use_categories):
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, self.hid_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hid_dim) x 32 x 32
            nn.Conv2d(self.hid_dim, self.hid_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hid_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (self.hid_dim*2) x 16 x 16
            nn.Conv2d(self.hid_dim*2, self.hid_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hid_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (self.hid_dim*4) x 8 x 8
            nn.Conv2d(self.hid_dim*4, self.hid_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hid_dim * 8),
            # state size (self.hid_dim * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        if self.dataset == 'pororo':
            self.get_cond_logits = D_GET_LOGITS(self.hid_dim, 128 + self.text_dim + 9 , video_len=1, dataset=self.dataset) # self.hid_dim, hD + tD + 9 . hD --> 128로 통일해야함.. (7/15)
        else:
            self.get_cond_logits = D_GET_LOGITS(self.hid_dim, self.text_dim, 1)
        self.get_uncond_logits = None

    def forward(self, image):
        """
        image : B, C, H, W
        """
        img_embedding = self.encode_img(image)

        return img_embedding
        

class D_STY(nn.Module):
    def __init__(self, hid_dim, text_dim=75, dataset='clevr'):
        super(D_STY, self).__init__()
        self.hid_dim = hid_dim
        self.dataset = dataset

        self.encode_img = nn.Sequential(
            nn.Conv2d(3, self.hid_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (self.hid_dim) x 32 x 32
            nn.Conv2d(self.hid_dim, self.hid_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hid_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size (self.hid_dim*2) x 16 x 16
            nn.Conv2d(self.hid_dim*2, self.hid_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hid_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size (self.hid_dim*4) x 8 x 8
            nn.Conv2d(self.hid_dim*4, self.hid_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hid_dim * 8),
            
            # state size (self.hid_dim * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        if dataset == 'pororo':
            self.get_cond_logits = D_GET_LOGITS(self.hid_dim, 128 + text_dim + 9, video_len=0, dataset=dataset)
        else:
            self.get_cond_logits = D_GET_LOGITS(self.hid_dim, text_dim, video_len=4)
        # self.get_uncond_logits = None
        # self.cate_classify = None

    def forward(self, story):
        # 원본코드 데이터인 경우 차원변환 해주기
        if story.shape[1] == 3:
            story = story.permute(0,2,1,3,4).contiguous() #B, C, L, H, W --> B, L, C, H, W
        N, video_len, C, W, H = story.shape 
        story = story.reshape(-1, C,W,H)
        story_embedding = torch.squeeze(self.encode_img(story))
        _, C1, W1, H1 = story_embedding.shape

        if self.dataset == 'pororo':
            story_embedding = story_embedding.view(N, video_len, C1, W1, H1)
            story_embedding = story_embedding.mean(1).squeeze() #B, self.dim*8, 4, 4
        else:            
            story_embedding = story_embedding.permute(2,3,0,1)
            story_embedding = story_embedding.view( W1, H1, N,video_len * C1)
            story_embedding = story_embedding.permute(2,3,0,1)
        return story_embedding
    

class STORYGAN(nn.Module):
    def __init__(self, text_dim=75, hid_dim=128, seq_len=4, lr_mul=1, dataset='clevr'):
        super().__init__()
        ngf = 1536
        self.hid_dim = hid_dim

        # 데이터셋에 따라 label num concat해줘야하는데.. 어떻게 분기해줘야 효율적인가.
        self.story_encoder = StoryEncoder(text_dim=text_dim, out_dim=hid_dim, seq_len=seq_len, lr_mul=1, dataset=dataset) 
        self.text_encoder = TextEncoder(text_dim=text_dim, hid_dim=hid_dim, seq_len=seq_len, dataset=dataset) 
        self.generator = Generator(ngf)

        
        self.make_g_input = nn.Sequential(
            nn.Linear(256, 15**2,bias=False),
            nn.BatchNorm1d(15**2)
        )
        

        # 아마도 무쓸모
        self.dfn = DynamicFilterLayer((15,15,1), pad=(7,7))
        self.filter_net = nn.Sequential(
            nn.Linear(256, 15**2,bias=False),
            nn.BatchNorm1d(15**2)
        )
        
    def make_img(self, input_desc, full_story):
        
        """
        1. clevr
        input_desc: B,tD
        full_story: B,L,tD

        2. 뽀로로
        input_desc : B, tD + 9
        full_story : B, L, tD
        """

        B,D = input_desc.shape
        z_code, mu, logvar = self.story_encoder(full_story) # B, hd 
        input_desc = input_desc.unsqueeze(1) #B, Td -> B, 1, Td
        out1 = self.text_encoder(input_desc ,z_code) #B,128, B128
        g_hiddens, _ = out1 # local한 정보 담고있음
        # h_hiddens, _ = out2 # global한 정보도 담고있음(by storyencoder) B, L, 128
        
        # h_hiddens = h_hiddens.view(-1, h_hiddens.shape[-1]) #B*L, 128
        g_hiddens = g_hiddens.view(-1, g_hiddens.shape[-1]) #B*L, 128
        
        g_input = self.make_g_input(g_hiddens)
        g_input = g_input.view(-1, 1, 15, 15)

        # filter = self.filter_net(g_hiddens)
        # filter = filter.view(-1, 1, 15, 15)
        # new_g_input = self.dfn([g_input, filter])

        fake_imgs = self.generator(g_input)             #B*L,3,64,64
        # fake_imgs = self.generator(new_g_input)             #B*L,3,64,64
        W = fake_imgs.shape[-1]
        fake_imgs = fake_imgs.view(B,-1,W,W)          #B,3,64,64
        return fake_imgs, mu, logvar
    
     

    def make_stories(self, input_seq, full_story):
        
        """
        - 원 코드와 비교
        input_seq == motion_input
        full_story == content_input

        1. Clevr
        input_Seq: B,L,tD
        full_story: B,L,tD ---> B, L * tD로 바꿀거임

        2. 뽀로로
        input_seq : B, L, tD + 9
        full_story : B, L, tD

        """
        B,L,_ = input_seq.shape

        z_code, mu, logvar = self.story_encoder(full_story) # BhD
        out1 = self.text_encoder(input_seq ,z_code) #BLhD, BhD
        g_hiddens, _ = out1 # local한 정보 담고있음
        # h_hiddens, _ = out2 # global한 정보 담고있음(by storyencoder)
        
        g_hiddens = g_hiddens.reshape(-1, g_hiddens.shape[-1])
        
        input_seq = input_seq.view(-1, input_seq.shape[-1])
        g_input = self.make_g_input(g_hiddens) #BL, hD
        g_input = g_input.view(-1, 1, 15, 15)

        fake_imgs = self.generator(g_input)             #B*L,3,64,64
        # fake_imgs = self.generator(new_g_input)             #B*L,3,64,64
        W = fake_imgs.shape[-1]
        fake_imgs = fake_imgs.view(B,L,-1,W,W)          #B,L,3,64,64
        return fake_imgs, mu, logvar
 