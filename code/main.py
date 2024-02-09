import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboard_logger import tb_logger
import os
import math
import argparse
import time 
from train_models import validate, validate_contrastive, train, train_contrastive
from models import SupCEResNet
from models import SupConLoss
from dataset import ImbalanceCIFAR, ImbalanceCIFAR10, ImbalanceCIFAR100
from util import MixupLTDataloader, save_model, adjust_learning_rate, set_optimizer
from focal import focal_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')

    # to run the custom dataset set the dataset = False, data_folder='...path'
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    # TODO: Check what default imabalance ratio should be.
    parser.add_argument('--imbalance_ratio', type=float, default=0.1)

    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'CE', 'Focal'], help='choose method')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight of the contrastive loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight of the contrastive loss')

    parser.add_argument('--mixup', type=bool, default=False)
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        # TODO: automatically pull the mean and std from a custom dataset.
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_model(opt):
    
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    return model

def main():
    best_acc = 0
    args = parse_option()

    # Define the dataset 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == 'cifar10':
        ds_train = ImbalanceCIFAR10(train=True, transform=transform, imbalance_ratio=args.imbalance_ratio)
        ds_test = ImbalanceCIFAR10(train=False, transform=test_transform, imbalance_ratio=1)  # imbalance_ratio=1 to keep original distribution

    elif args.dataset == 'cifar100':
        # Initialize the datasets
        ds_train = ImbalanceCIFAR100(train=True, transform=transform, imbalance_ratio=args.imbalance_ratio)
        ds_test = ImbalanceCIFAR100(train=False, transform=test_transform, imbalance_ratio=1)  # imbalance_ratio=1 to keep original distribution
    else: 
        # custom dataset /bishoy/desktop/bird-dataset /train /test
        ds_train = ImbalanceCIFAR(train=True, transform=None, imbalance_ratio=args.imbalance_ratio, dataset_Path = f"{args.data_folder}/train", debug = False)
        ds_test = ImbalanceCIFAR(train=False, transform=None, imbalance_ratio=1, dataset_Path = f"{args.data_folder}/test", debug = False)
        # TODO mean, std = calculate_mean_std(ds_train)
        # this is in util.py

    
    # TODO
    # Create testing DataLoaders
    # test_loader_birdDataset = DataLoader(Imbalanced_Bird_Dataset_test, batch_size=64, shuffle=False)

    if args.loss == 'CE':
        # Define the crossentropy loss
        criterion = torch.nn.CrossEntropyLoss() 
    elif args.loss == 'SupCon': 
        # Define the contrastive loss 
        criterion = SupConLoss(temperature=args.temp)
    elif args.loss == 'Focal':
        # TODO
        print('Not implemented yet')
        raise NotImplementedError('Gotta do this one.')
        # Define the focal loss 
        # Define the weight tensor
        # weights_tensor = focal_weights(ds_train)
        # criterion = focal_loss(alpha=weights_tensor, gamma=2.0, reduction='mean')
    else: 
        raise ValueError('Loss not configured.')
    
    # Define the dataloaders
    if args.mixup: 
        # TODO: Import mixup from util.py
        train_loader = MixupLTDataloader(dataset=ds_train, batch_size= args.batch_size)
    else:    
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Create validation DataLoaders
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    # build model and criterion
    model, criterion = set_model(args) # TODO something not updated for focal , use_ce

    # build optimizer
    optimizer = set_optimizer(args, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if args.loss == 'SupCon': 
            loss, train_acc = train_contrastive(train_loader, model, criterion, optimizer, epoch, args, ALPHA=args.alpha, BETA=args.beta)

        else:
            loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
            
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        if args.loss == 'SupCon': 
            loss, val_acc = validate_contrastive(test_loader, model, criterion, args, epoch, ALPHA=args.alpha, BETA=args.beta)
        else:
            loss, val_acc = validate(test_loader, model, criterion, optimizer, epoch, args)
            
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        # TODO: on the highest accuracy, save the file
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        args.save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()