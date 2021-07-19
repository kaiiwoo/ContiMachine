# -*- coding: utf-8 -*- 
from PIL import Image
import torchvision.transforms as transforms
import argparse
from researcher import Researcher
import torch
import os
import functools
import numpy as np 

# 빨리 통합해야됨 ㅈㄴ 헷갈림...
class HyperParams(object):
    data_type='pororo'
    num_gpu = 1
    latent_dim = 10
    img_size = 64
    num_workers = 2
    video_len = 4
    vis_count = 10 #
    label_num = 15 # 

    # image_batch_size / story_batch_size 나눠줘야되나
    batch_size = 24
    max_epoch = 120
    lr_decay_epoch = 20
    snapshot_interval = 10
    D_lr = 2e-4
    G_lr = 2e-4
    kl_coeff = 1.0
    text_dim = 75 #75 - clevr, 256 - pororro
    hid_dim = 96 #96 - clevr, 128 / 124 - pororo
    n_channels= 64
    # df_dim = 96
    # gf_dim = 192
    
    alpha = 1
    beta = 1


def main(args):
    image_transform = transforms.Compose([
    Image.fromarray,
    transforms.Resize((args.img_size, args.img_size)), #(64, 64)
    transforms.ToTensor(),
    lambda x : x[:3, ::],
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

    if args.dataset=='clevr':
        print("StoryCLIP - train with Clevr Dataset")

        image_transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize((args.img_size, args.img_size)), #(64, 64)
            transforms.ToTensor(),
            lambda x : x[:3, ::],
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                ])


        from data_utils.clevr_utils import ImageData, StoryData, get_dataloader
        data_dir = './clevr_dataset'
        imagedata = ImageData(data_dir, image_transform)
        storydata = StoryData(data_dir, image_transform)
        testdata = StoryData(data_dir, image_transform)
        
        image_loader = get_dataloader(imagedata, args.batch_size, args.num_workers, shuffle=True)
        story_loader = get_dataloader(storydata, args.batch_size, args.num_workers, shuffle=True)
        test_loader = get_dataloader(testdata, args.batch_size, args.num_workers, shuffle=False)

    if args.dataset=='pororo':
        print("StoryCLIP - train with Pororo Dataset")

        image_transform = transforms.Compose([
            Image.fromarray,
            transforms.Resize((args.img_size, args.img_size)), #(64, 64)
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ])

        from data_utils.pororo_utils import VideoFolderDataset, StoryDataset, ImageDataset

        def video_transform(video, image_transform):
            vid = []
            for im in video:
                vid.append(image_transform(im))

            vid = torch.stack(vid).permute(1, 0, 2, 3)
            return vid

        dir_path = '/home/keonwookim/StoryGAN/StoryGAN_pororo/img_pororo_eccv/'
        output_dir = os.path.join(args.output_dir, 'pororo')

        video_len = 5
        n_channels = 3
        video_transforms = functools.partial(video_transform, image_transform=image_transform)

        counter = np.load(dir_path + 'frames_counter.npy', allow_pickle=True).item()
        base = VideoFolderDataset(dir_path, counter = counter, cache = dir_path, min_len = 4, train=True)
        storydataset = StoryDataset(base,dir_path, video_transforms)
        imagedataset = ImageDataset(base, dir_path, image_transform)

        image_loader = torch.utils.data.DataLoader(
            imagedataset, batch_size=args.batch_size * args.num_gpu,
            drop_last=True, shuffle=True, num_workers=int(args.num_workers))

        story_loader = torch.utils.data.DataLoader(
            storydataset, batch_size=args.batch_size * args.num_gpu,
            drop_last=True, shuffle=True, num_workers=int(args.num_workers))

        # Test set prepare 
        base_test = VideoFolderDataset(dir_path, counter = counter, cache = dir_path, min_len = 4, train=False)
        testdataset = StoryDataset(base_test, dir_path, video_transforms)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=24, drop_last=True, shuffle=False, num_workers=int(args.num_workers))


    researcher = Researcher(args, image_loader, story_loader)
    researcher.train()

if __name__ =="__main__":
    hp = HyperParams()
    parser = argparse.ArgumentParser()
    
    # 하이퍼파라미터는 아니지만 필요한 세팅
    parser.add_argument('--dataset', type=str, default=hp.data_type, choices = ['pororo', 'clevr'], help='Data type you want to train')
    parser.add_argument('--mode', type=str, default='train', help='choose Train or Test')
    parser.add_argument('--output_dir', type=str, default='./my_result', help='choose Train or Test')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpu to use')
    parser.add_argument('--num_workers', type=int, default=hp.num_workers, help='number of workers to use')
    
    
    parser.add_argument('--num_gpu', type=int, default=hp.num_gpu)
    
    parser.add_argument('--latent_dim', type=int, default=hp.latent_dim)
    parser.add_argument('--img_size', type=int, default=hp.img_size)
    parser.add_argument('--video_len', type=int, default=hp.video_len)
    parser.add_argument('--vis_count', type=int, default=hp.vis_count)
    parser.add_argument('--label_num', type=int, default=hp.label_num)
    parser.add_argument('--batch_size', type=int, default=hp.batch_size)
    parser.add_argument('--max_epoch', type=int, default=hp.max_epoch)
    parser.add_argument('--lr_decay_epoch', type=int, default=hp.lr_decay_epoch)
    parser.add_argument('--snapshot_interval', type=int, default=hp.snapshot_interval)
    parser.add_argument('--num_channels', type=int, default=64)
    
    # 학습관련 하이퍼파라미터
    parser.add_argument('--alpha', type=float, default=hp.alpha)
    parser.add_argument('--beta', type=float, default=hp.beta)
    
    parser.add_argument('--D_lr', type=float, default=hp.D_lr)
    parser.add_argument('--G_lr', type=float, default=hp.G_lr)
    parser.add_argument('--kl_coeff', type=float, default=hp.kl_coeff)
    
    parser.add_argument('--text_dim', type=int, default=hp.text_dim)
    parser.add_argument('--hid_dim', type=int, default=hp.hid_dim)
    parser.add_argument('--n_channels', type=int, default=hp.n_channels)
    # parser.add_argument('--df_dim', type=int, default=hp.df_dim)
    # parser.add_argument('--gf_dim', type=int, default=hp.gf_dim)
    

    args = parser.parse_args()
    main(args)
    