import argparse
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import ImbalanceCIFAR, ImbalanceCIFAR10, ImbalanceCIFAR100
from focal import focal_loss, focal_weights
from losses import SupConLoss
from models import SupCEResNet, SupConResNet
from train_models import validate, validate_contrastive, train, train_contrastive
from util import save_model, adjust_learning_rate, set_optimizer, plot_loss_curves, \
    plot_accuracy_curves, TwoCropTransform, evaluate_model_performance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--image_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs')
    # parser.add_argument('--n_cls', type=int, default=525, help='number of classes in the dataset'),
    parser.add_argument('--load_dataset', type=int, default=1, help='Load dataset from the .pt files'),
    parser.add_argument('--debug', type=int, default=0, help='Use the debug mode.'),

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
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

    parser.add_argument('--mixup', type=int, default=0)
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
    # check 02/23/2024
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        # TODO: automatically pull the mean and std from a custom dataset
        assert opt.data_folder is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save_models'  # .format(opt.dataset)
    opt.tb_path = './save_tensorboard'  # .format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_alpa_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial, opt.alpha)

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
    if opt.method == 'SupCon':
        model = SupConResNet(name=opt.model, num_classes=opt.n_cls)
    else:
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)  # changed here
        cudnn.benchmark = True

    return model


def norm(x):
    return x / 255.0


def main():
    best_acc = 0
    args = parse_option()

    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),  # rotates the image by up to 10 degrees
        transforms.ToTensor(),
        norm
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    test_transform = transforms.Compose([

        transforms.ToTensor(),
        norm
    ])
    print('Loading dataset!')
    print('Dataset', args.dataset)
    print('Data folder', args.data_folder)
    print('Use existing dataset', args.load_dataset)

    ds_train = None

    ds_test = None

    if args.load_dataset:
        ds_train = None
        ds_test = None

        if args.dataset == 'cifar10':
            if args.method == 'SupCon':
                ds_train = torch.load(ds_train, 'supcon_cifar10_train.pt')
                ds_test = torch.load(ds_test, 'supcon_cifar10_test.pt')
            else:
                ds_train = torch.load(ds_train, 'cifar10_train.pt')
                ds_test = torch.load(ds_test, 'cifar10_test.pt')

            args.n_cls = 10
        elif args.dataset == 'cifar100':
            if args.method == 'SupCon':
                ds_train = torch.load(ds_train, 'supcon_cifar100_train.pt')
                ds_test = torch.load(ds_test, 'supcon_cifar100_test.pt')
            else:
                ds_train = torch.load(ds_train, 'cifar100_train.pt')
                ds_test = torch.load(ds_test, 'cifar100_test.pt')

            args.n_cls = 100
        else:
            # TODO: Update based on what you end up calling the supcon files.
            ds_train = torch.load('./train.pt')
            ds_test = torch.load('./test.pt')
            args.n_cls = 75
    else:
        if args.dataset == 'cifar10':
            if args.method == 'SupCon':
                print('Dataset is cifar10')
                ds_test = ImbalanceCIFAR10(train=False, transform=TwoCropTransform(test_transform), imbalance_ratio=1,
                                           debug=args.debug)
                ds_train = ImbalanceCIFAR10(train=True, transform=TwoCropTransform(train_transform),
                                            imbalance_ratio=args.imbalance_ratio, debug=args.debug)

                torch.save(ds_train, 'supcon_cifar10_train.pt')
                torch.save(ds_test, 'supcon_cifar10_test.pt')
            else:

                print('Dataset is cifar10')
                ds_train = ImbalanceCIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio,
                                            debug=args.debug)

                ds_test = ImbalanceCIFAR10(train=False, transform=test_transform,
                                           imbalance_ratio=1,
                                           debug=args.debug)  # imbalance_ratio=1 to keep original distribution
                torch.save(ds_train, 'cifar10_train.pt')
                torch.save(ds_test, 'cifar10_test.pt')
            args.n_cls = 10

        elif args.dataset == 'cifar100':
            # Initialize the datasets
            if args.method == 'SupCon':
                print('Dataset is cifar100')
                ds_test = ImbalanceCIFAR100(train=False, transform=TwoCropTransform(test_transform), imbalance_ratio=1,
                                            debug=args.debug)
                ds_train = ImbalanceCIFAR100(train=True, transform=TwoCropTransform(train_transform),
                                             imbalance_ratio=args.imbalance_ratio, debug=args.debug)

                torch.save(ds_train, 'supcon_cifar100_train.pt')
                torch.save(ds_test, 'supcon_cifar100_test.pt')
            else:

                print('Dataset is cifar100')
                ds_train = ImbalanceCIFAR100(train=True, transform=train_transform,
                                             imbalance_ratio=args.imbalance_ratio,
                                             debug=args.debug)
                ds_test = ImbalanceCIFAR100(train=False, transform=test_transform,
                                            imbalance_ratio=1,
                                            debug=args.debug)  # imbalance_ratio=1 to keep original distribution
                torch.save(ds_train, 'cifar100_train.pt')
                torch.save(ds_test, 'cifar100_test.pt')
            args.n_cls = 100
        else:
            test_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                norm
            ])

            train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),  # rotates the image by up to 10 degrees
                transforms.ToTensor(),
                norm
            ])

            # custom dataset /bishoy/desktop/bird-dataset /train /test
            if args.method == 'SupCon':
                print('Dataset is ButterflyDataset')
                ds_test = ImbalanceCIFAR(train=False, transform=TwoCropTransform(test_transform), imbalance_ratio=1,
                                         dataset_Path=f"{args.data_folder}/test", debug=args.debug)
                ds_train = ImbalanceCIFAR(train=True, transform=TwoCropTransform(train_transform),
                                          imbalance_ratio=args.imbalance_ratio,
                                          dataset_Path=f"{args.data_folder}/train", debug=args.debug)
            else:

                print('Dataset is ButterflyDataset')

                # TODO change the valid to train and test
                print(args.data_folder)
                train_dataset_Path = os.path.join(args.data_folder, "train")  # \ /
                test_dataset_Path = os.path.join(args.data_folder, "test")  # \ /

                print('Train dataset path', train_dataset_Path)
                print('Test dataset path', test_dataset_Path)

                ds_train = ImbalanceCIFAR(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio,
                                          dataset_Path=train_dataset_Path, debug=args.debug)
                ds_test = ImbalanceCIFAR(train=False, transform=test_transform,
                                         imbalance_ratio=1, dataset_Path=test_dataset_Path,
                                         debug=args.debug)  # imbalance_ratio=1 to keep original distribution

            args.n_cls = 75
            print('Saving the imbalanced datasets!')
            torch.save(ds_train, 'train.pt')
            torch.save(ds_test, 'test.pt')

    print('Defining the loss.')
    if args.method == 'CE':
        # Define the crossentropy loss
        criterion = torch.nn.CrossEntropyLoss()
    elif args.method == 'SupCon':
        # Define the contrastive loss 
        criterion = SupConLoss(temperature=args.temp)
    elif args.method == 'Focal':
        print('DS train', type(ds_train))
        weights_tensor = focal_weights(ds_train)
        criterion = focal_loss(alpha=weights_tensor, gamma=2.0, reduction='mean')
        criterion = criterion.to(device)
    else:
        raise ValueError('Loss not configured.')

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Create validation DataLoaders
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # build model and criterion
    model = set_model(args)  # TODO something not updated for focal , use_ce

    # build optimizer
    optimizer = set_optimizer(args, model)

    print('Starting training.')

    # changed heere for early stopping
    best_val_loss = math.inf  # Monitor the best validation accuracy
    patience = 10  # How many epochs to wait after last time validation accuracy improved.
    patience_counter = 0  # Counter for how many epochs have gone by without improvement

    # Initialize lists for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    model_dir = None

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # Train for one epoch
        time1 = time.time()
        if args.method == 'SupCon':
            loss, train_acc = train_contrastive(train_loader, model, criterion, optimizer, epoch, args,
                                                ALPHA=args.alpha, BETA=args.beta)
        else:
            loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        time2 = time.time()

        train_losses.append(loss)
        train_accuracies.append(train_acc)
        print('epoch {}, accuracy: {:.2f}, total time {:.2f}'.format(epoch, train_acc, time2 - time1))

        # Evaluation
        if args.method == 'SupCon':
            val_loss, val_acc = validate_contrastive(test_loader, model, criterion, args, epoch, ALPHA=args.alpha,
                                                     BETA=args.beta)
            # changed here 02/15/2024
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print('val_loss', val_loss, epoch)
            print('val_acc', val_acc, epoch)
        else:

            val_loss, val_acc = validate(test_loader, model, criterion, args)
            # changed here 02/15/2024
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print('val_loss', val_loss, epoch)
            print('val_acc', val_acc, epoch)

        # Check if current validation accuracy is the best we've seen so far
        if val_loss < best_val_loss:
            # Save the model if validation accuracy has improved
            best_model_file = os.path.join(args.save_folder, 'best_model.pth')
            model_dir = save_model(model, optimizer, args, epoch, best_model_file)

            print(f"Validation Loss improved from {best_val_loss} to {val_loss}. Saving model to {best_model_file}")
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
        else:
            print(f'Val loss of {val_loss: .3f} is not better than {best_val_loss: .3f}')
            patience_counter += 1  # Increment patience counter
            print(f"Validation Loss did not improve. Patience counter: {patience_counter}/{patience}")

        # Early stopping check
        if patience_counter >= patience:
            print(f"Stopping early due to no improvement in validation Loss for {patience} epochs")
            break

    """
    Load the best model 
    """
    inf_model = set_model(args)
    # Load the saved file
    checkpoint = torch.load(best_model_file)

    # Update model and optimizer states
    inf_model.load_state_dict(checkpoint['state_dict'])

    evaluate_model_performance(inf_model, test_loader, args, device, model_dir,
                               save_path='confusion_matrix.png')  # Add this line
    plot_loss_curves(train_losses, val_losses, model_dir, title='Training and Validation Loss', xlabel='Epochs',
                     ylabel='Loss',
                     save_path='loss_curves.png')
    plot_accuracy_curves(train_accuracies, model_dir, val_accuracies, title='Training and Validation Accuracy',
                         xlabel='Epochs', ylabel='Accuracy', save_path='accuracy_curves.png')
    print('best Loss: {:.2f}'.format(best_val_loss))
    print('model directory is' + model_dir)


if __name__ == '__main__':
    main()
