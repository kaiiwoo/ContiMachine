


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