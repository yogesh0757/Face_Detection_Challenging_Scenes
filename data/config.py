# config.py
cfg_re50 = {
    'min_sizes': [[11, 16, 22], [44, 65, 87], [174, 262, 348]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 4,
    'epoch': 140,
    'decay1': 100,
    'decay2': 120,
    'image_size': 640,
    'pretrain': True,
    'return_layers2': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 256,
    'out_channel': 256
}
