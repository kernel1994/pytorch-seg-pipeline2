import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# random seed
random_state = 7
torch.manual_seed(random_state)

import pathlib
# data path
data_dir = pathlib.Path('/path/to/dataset')
# ori path
ori_dir = data_dir.joinpath('ori')
# seg path
seg_dir = data_dir.joinpath('seg')
# results save path
main_dir = pathlib.Path('/path/to/results')

# train or test phase
is_train = True
# number KFold split
n_kfold = 5
# model name
model_name = 'unet'

# heigh, width, channel
h = 80
w = 80
c = 1
# number of seg class
n_class = 2
# number of samples used. None for all
n_samples = None
# test size
test_size = 10
# batch size
batch_size = 4
# train epoch
max_epoch = 10
# learning rate
lr = 1e-3
milestones = [100]
gamma = 0.9
drop_out = 0.2

# best model path
best_model_path_name = 'best_model.pth'
# LeaveOneOut dice summary
sub_summary_name = 'sub_summary.csv'
summary_name = 'summary.csv'

# EvaluateSegmentation path
evaluator = '/path/to/EvaluateSegmentation'
