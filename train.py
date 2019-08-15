import argparse
import torch
from dataset_factory import dataset_factory
from model import SkynetModel
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from experimentlogger import Experiment
from tqdm import tqdm

def argparser():
    parser = argparse.ArgumentParser(description='Skynet Model')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-mode', type=str, default='full', help="Define the model mode, either 'full', 'images-only', 'features-only' or 'data-driven'")
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no-data-augment',action='store_true', default=False,
                        help='disables data augmentation')
    parser.add_argument('--data-augmentation-angle',type=float, default=20)
    parser.add_argument('--out-channels-l1', type=int, default=200)
    parser.add_argument('--offset-811', type=int, default=13)
    parser.add_argument('--offset-2630', type=int, default=-4)
    
    args = parser.parse_args()
    return args


def run(args):
    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    if args.cuda:
        print('CUDA enabled')
        torch.cuda.manual_seed(args.seed)


    
    if not args.model_mode == 'features-only':
        args.use_images = True
        print("Using images.")
    else:
        args.use_images = False

    

    if args.no_data_augment:
        transform = False
    else:
        transform = True
        print("Using data augmentation")
    
    num_workers = 4
    
    # Load data
    train_dataset, test_dataset = dataset_factory(use_images=args.use_images, transform=transform, data_augment_angle=args.data_augmentation_angle) # No image folder means loading from hdf5 file
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers, drop_last=False, shuffle=False)

    # Instansiate model
    args.num_features = train_dataset.features.shape[1]+1
    args.image_size = [256, 256]
    args.out_channels = [int(args.out_channels_l1), 100, 50, 25, 12, 1]
    args.kernel_size = [(5,5), (3,3), (3,3), (3,3), (2,2), (2,2)]
    args.nn_layers = [200, 200]
    args.channels = 1

    rsrp_mu = train_dataset.target_mu
    rsrp_std = train_dataset.target_std
   

    model = SkynetModel(args, rsrp_mu = rsrp_mu, rsrp_std = rsrp_std)
    if args.cuda:
        model.cuda()
    
    # Define loss function, optimizer and LR scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_model = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training loop
    train_loss = []
    test_loss = []
    def train(epoch):
        # Called by the loop
        trainloss = 0
        with tqdm(total = len(train_loader)) as pbar:
            for idx, (feature, image, target, dist) in enumerate(train_loader):
                if args.cuda:
                    image = image.cuda()
                    feature = feature.cuda()
                    target = target.cuda()
                    dist = dist.cuda()

                optimizer.zero_grad()

                output, sum_output = model(feature, image, dist)

                loss = criterion(sum_output, target)
                loss.backward()
                trainloss += loss.item()
                optimizer.step()
                pbar.update(1)

            train_loss.append(trainloss/idx)

        pbar.close()
        
    def test(epoch):
        # Called by the loop
        testloss = 0
        with torch.no_grad():
            with tqdm(total = len(test_loader)) as pbar:
                for idx, (feature, image, target, dist) in enumerate(test_loader):
                    if args.cuda:
                        image = image.cuda()
                        feature = feature.cuda()
                        target = target.cuda()
                        dist = dist.cuda()
                    
                    output, sum_output = model(feature, image, dist)

                    loss = criterion(sum_output, target)
                    testloss += loss.item()
                    pbar.update(1)

                test_loss.append(testloss/idx)
            pbar.close()

    for epoch in range(args.epochs):
        model.train()
        train(epoch)
        model.eval()
        test(epoch)
        scheduler_model.step(test_loss[-1])
        print("Epoch: {}, train_loss: {}, test_loss: {}".format(epoch, train_loss[-1], test_loss[-1]))

        if optimizer.param_groups[0]['lr'] < 1e-7:
            print('Learning rate too low. Early stopping.')
            break


    exp = Experiment('file', config=args.__dict__, root_folder='exps/')
    results_dict = dict()
    results_dict['train_loss'] = train_loss
    results_dict['test_loss'] = test_loss
    exp.results = results_dict
    exp.save()

    torch.save(model.state_dict(), exp.root_folder+'/models/{}_model_{:.3f}.pt'.format(exp.id,test_loss[-1]))



if __name__ == '__main__':
    args = argparser()
    run(args)
    