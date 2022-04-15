from easydict import EasyDict

cfg = EasyDict()
cfg.batch_size = 1
cfg.epochs = 1
cfg.in_cannel = 3
cfg.n_classes = 3
cfg.learning_rate = 1e-2
# cfg.learning_rate = 1e-5
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 25
cfg.inference_threshold = 0.75

cfg.transunet = EasyDict()
cfg.transunet.img_dim = 512
cfg.transunet.out_channels = 128
cfg.transunet.head_num = 4
cfg.transunet.mlp_dim = 512
cfg.transunet.block_num = 8
cfg.transunet.patch_dim = 16


cfg.my = EasyDict()
cfg.my.unet_epoch = 1
cfg.my.unet_batch_size = 1
cfg.my.trans_epoch = 100
cfg.my.trans_batch_size = 1
cfg.my.trans_img_dim = 512
cfg.my.trans_out_channels = 128
cfg.my.trans_head_num = 4
cfg.my.trans_mlp_dim = 512
cfg.my.trans_block_num = 8
cfg.my.trans_patch_dim = 16
