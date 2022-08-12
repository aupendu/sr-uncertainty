import argparse

parser = argparse.ArgumentParser(description='Training of SOTA SISR Algorithms')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=1, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--BI_input', action='store_true', help='Bicubic Input')
parser.add_argument('--save_MV', action='store_true', help='save mean & variances for different batches')

# Data specifications
parser.add_argument('--HRpath', type=str, default='/home/aupendu/ISR/data/HR/', help='High-Resolution data path')
parser.add_argument('--LRpath', type=str, default='/home/aupendu/ISR/data/Basic/', help='Low-Resolution data path')
parser.add_argument('--data_train', type=str, help='Train dataset name')
parser.add_argument('--data_val', type=str, help='Valid dataset name')
parser.add_argument('--scale', type=str, default='', help='Super-resolution scale')
parser.add_argument('--lr_patch_size', type=int, help='Input patch size')
parser.add_argument('--n_colors', type=int, default=3, help='Number of color channels to use')
parser.add_argument('--rgb_range', type=float, help='range of image intensity')

# Model specifications
parser.add_argument('--model', default='RDN', help='model name')
parser.add_argument('--trained_model', type=str, help='pre-trained model directory')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--pre_train', action='store_true', help='Use pretrain model')
parser.add_argument('--only_MV', action='store_true', help='Use pretrain model')
parser.add_argument('--test_data_val', type=str, help='Valid dataset name')

# Training specifications
parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--batch_size_val', type=int, default=1, help='input batch size for validation/testing')
parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
parser.add_argument('--decay', type=str, default='200+400+600+800', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')

args = parser.parse_args()

