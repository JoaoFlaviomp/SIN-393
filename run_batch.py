# Author: João Fernando Mari
# joaofmari.github.io
# https://github.com/joaofmari

import os
import time
import datetime
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--ds', help='Dataset name.', type=str, default='Scoliosis')
parser.add_argument('--ds_split', help='How ds is split: ["train-val-test", "train-test", "no-split"].', type=str, default='no-split')

# Adicionando argumentos para técnicas de aumento de dados
parser.add_argument('--cutmix', help='Use CutMix augmentation.', action='store_true')
parser.add_argument('--cutout', help='Use Cutout augmentation.', action='store_true')
parser.add_argument('--mixup', help='Use MixUp augmentation.', action='store_true')
parser.add_argument('--cutmix_alpha', help='Alpha parameter for CutMix.', type=float, default=1.0)
parser.add_argument('--cutout_n_holes', help='Number of holes for Cutout.', type=int, default=1)
parser.add_argument('--cutout_length', help='Length of holes for Cutout.', type=int, default=128)
parser.add_argument('--mixup_alpha', help='Alpha parameter for MixUp.', type=float, default=1.0)

args = parser.parse_args()

EXP_PATH_MAIN = f'exp_{args.ds}'

# TTA strategy
dapred = 0

arch_list = ['resnet50',
             'vit_b_16', 
             'swin_v2_t',
             #'efficientnet_v2_s',
             # Incluir mais arquiteturas...
            ]

# Estratégias de aumento de dados. Verificar arquivo 'data_aug_3.py'
datrain_list = [0,2] 
daval_list = [0,0]
datest_list = [0,0]

# Total de épocas de treinamento.
epochs = 200 # 200 # 200
num_workers = 0
s_seed = 70

# Replace by the best hyperparameter values:
bs_dict = {'resnet50' : 64, 
           'vit_b_16' : 64, 
           'swin_v2_t' : 64,
           #'efficientnet_v2_s' : 64,
           # Para outras arquiteturas
           } 

lr_dict = {'resnet50' : 0.0001, 
           'vit_b_16' : 0.0001,
           'swin_v2_t' : 0.0001,
           #'efficientnet_v2_s' : 0.0001,
           # Para outras arquiteturas
           } 

scheduler = 'cossine'

if scheduler == 'steplr':
    ss_ = 40
else:
    ss_ = 0


da2_list = ['none', 'cutmix', 'cutout', 'mixup']

ec = 0
for arch in arch_list:
    # ec = 0
    for datrain, daval, datest in zip(datrain_list, daval_list, datest_list):

        for da2 in da2_list:

            cmd_str = f'nohup python train_model.py --ds {args.ds} --ds_split {args.ds_split} ' + \
                    f'--arch {arch} --num_workers {num_workers} --scheduler {scheduler} ' + \
                    f' --ss {ss_} --es --s_seed {s_seed} ' + \
                    f'--bs {bs_dict[arch]} --lr {lr_dict[arch]} --ep {epochs} --optimizer "Adam" ' + \
                    f'--datrain {datrain} --daval {daval} --datest {datest} --ec {ec} ' + \
                    f'--da2 {da2}' # + \
                    #f'--cutmix_alpha {args.cutmix_alpha} --cutout_n_holes {args.cutout_n_holes} ' + \
                    #f'--cutout_length {args.cutout_length} --mixup_alpha {args.mixup_alpha} '

            print(cmd_str)
            ec = ec + 1

            os.system(cmd_str)

if os.path.exists('./nohup.out'):
    suffix = ''
    while True:
        if os.path.exists(os.path.join(EXP_PATH_MAIN, 'nohup' + suffix + '.out')):
            suffix += '_'
        else:
            break
    shutil.move('./nohup.out', os.path.join(EXP_PATH_MAIN, 'nohup' + suffix + '.out'))

print('Done! (run_batch)')
