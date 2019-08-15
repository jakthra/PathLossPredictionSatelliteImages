import argparse
import torch
from dataset_factory import dataset_factory
from model import SkynetModel
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from experimentlogger import load_experiment
from easydict import EasyDict as edict
import os
import seaborn as sns
import numpy as np

def argparser():
    parser = argparse.ArgumentParser(description='Skynet Model')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--name', type=str, help='Name of experiment/model to load')
    parser.add_argument('--exp-folder', type=str, default='exps')
    args = parser.parse_args()
    return args


def run(args):
    cuda = not args.no_cuda and torch.cuda.is_available()


    

    if cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    if cuda:
        print('CUDA enabled')
        torch.cuda.manual_seed(args.seed)


    # Load data

    # Load experiment

    exp_root_path = args.exp_folder+"/"

    exp = load_experiment(args.name, root_path = exp_root_path)
    name = args.name
    args = edict(exp.config)
    args.name = name
    args.cuda = cuda
    args.data_augmentation_angle = 20
    # compatibility 
    if not 'offset_811' in args:
        args.offset_811 = 18
    
    if not 'offset_2630' in args:
        args.offset_2630 = 0
    
    train_dataset, test_dataset = dataset_factory(use_images=args.use_images, use_hdf5=True, transform=True, data_augment_angle=args.data_augmentation_angle)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=False)
    print(len(test_loader))


    rsrp_mu = train_dataset.target_mu
    rsrp_std = train_dataset.target_std
   

    model = SkynetModel(args, rsrp_mu = rsrp_mu, rsrp_std = rsrp_std)
 
    if args.cuda:
        model.cuda()

    # Find model name
    list_of_files = os.listdir('{}models/'.format(exp_root_path)) #list of files in the current directory
    for each_file in list_of_files:
        if each_file.startswith(args.name):  
            name = each_file

    
            

    model.load_state_dict(torch.load('{}models/{}'.format(exp_root_path, name)))
    model.eval()
    criterion = nn.MSELoss()
    MSE_loss_batch = 0
    with torch.no_grad():
        for idx, (feature, image, target, dist) in enumerate(test_loader):
            if args.cuda:
                image = image.cuda()
                feature = feature.cuda()
                target = target.cuda()
                dist = dist.cuda()

            correction_, sum_output_ = model(feature, image, dist)
            P  = model.predict_physicals_model(feature, dist)

            MSE_loss_batch += criterion(sum_output_, target)
            try:
                p = torch.cat([p, P], 0)
            except:
                p = P
            
            
            try:
                correction = torch.cat([correction, correction_],0)
            except:
                correction = correction_

            try:
                sum_output = torch.cat([sum_output, sum_output_],0)
            except:
                sum_output = sum_output_

            try:
                features = torch.cat([features, feature],0)
            except:
                features = feature

    # Check if folder with name in results exist
    results_folder_path = 'results/{}'.format(args.name)
    if not os.path.exists(results_folder_path):
        os.mkdir(results_folder_path)

    # Store predictions
    np.save(results_folder_path+"/correction.npy", correction)
    np.save(results_folder_path+"/sum_output.npy", correction)
    np.save(results_folder_path+"/pathloss_model.npy", correction)





if __name__ == '__main__':
    args = argparser()
    run(args)
    