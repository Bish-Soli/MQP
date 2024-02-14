import torch
import numpy as np
import torch.optim as optim
import numpy as np
import math

from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MixupLTDataloader():
    def __init__(self, batch_size, dataset, alpha=0.2):
        self.n_samples = len(dataset)
        self.n_classes = dataset.num_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_iterations = math.ceil(self.n_samples / batch_size)
        self.labels = np.array(dataset.targets)
        self.dataset = dataset

    def __len__(self):
        return self.n_samples
    def __iter__(self):
        self.iteration_index = 0
        return self
    
    def __next__(self):
        if self.iteration_index < self.n_iterations:
            self.iteration_index += 1
            return self.generate_batch()
        else:
            raise StopIteration

    def random_class(self):
        return np.random.randint(self.n_classes)

    def random_sample(self, random_class):
        # Randomly get index from the potential indices in the class
        indices = np.where(self.labels == random_class)[0] # this is a numpy array
        rand_idx = np.random.randint(len(indices)+1)
        rand_sample, rand_label = self.dataset.__getitem__(rand_idx)
        # print(type(rand_sample))
        return rand_sample, rand_label

    def generate_sample(self):
        rand_class = self.random_class()
        return self.random_sample(rand_class)

    def generate_mixup(self):
        # Define two samplers
        transform_to_tensor = ToTensor()  # Instantiate the transform
        probability = torch.zeros(self.n_classes)

        x1, y1 = self.generate_sample()
        x2, y2 = self.generate_sample()

        # Convert PIL Images to tensors
        x1 = transform_to_tensor(x1)
        x2 = transform_to_tensor(x2)

        x_hat = self.alpha * x1 + (1 - self.alpha) * x2
        y_hat = self.alpha * y1 + (1 - self.alpha) * y2

        c1 = math.floor(y_hat)
        decimal = y_hat - c1
        if c1 != self.n_classes - 1:
            c2 = c1 + 1
        probability[c1] = 1 - decimal
        probability[c2] = decimal
        return x_hat, torch.FloatTensor(probability)

    def generate_batch(self):
        data = []
        labels = []
        while len(data) < self.batch_size:
            x, y = self.generate_mixup()
            data.append(x)
            labels.append(y)

        return torch.stack(data), torch.stack(labels)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# TODO top1, top5 accuracy

# def accuracy(output, target, topk=(1, 5)):
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#
#         # Reshape and expand target to match pred shape
#         correct = pred.eq(target.view(1, -1).expand(batch_size, maxk))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0)
#             res.append(correct_k.mul_(100.0 / batch_size))
#
#         return res

import torch

def accuracy(output, target):
    with torch.no_grad():
        # Convert one-hot encoded vectors to class indices
        output_classes = torch.argmax(output, dim=1)
        target_classes = torch.argmax(target, dim=1)

        # Calculate the number of instances where predictions match the target
        correct = torch.sum(output_classes == target_classes)

        # Calculate the accuracy
        acc = correct.float() / target.size(0)

        return acc.item() * 100  # Convert to percentage





def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

    

def calculate_mean_std(ds_train):
    # TODO: Define function to calculate the mean and standard deviation across the training dataset
    """
    Should return the mean and std
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    Maybe you don't pass anything for the transform initially and then you update it
    """
    sum_rgb = np.array([0.0, 0.0, 0.0])
    sum_squared_rgb = np.array([0.0, 0.0, 0.0])
    total_pixels = 0
    
    for data, _ in ds_train:
        # Assuming data is a PIL Image, convert it to a numpy array
        numpy_data = np.array(data) / 255.0  # Scale pixel values to [0, 1]
        # Sum up pixel values and squared pixel values
        sum_rgb += numpy_data.sum(axis=(0, 1))
        sum_squared_rgb += (numpy_data ** 2).sum(axis=(0, 1))
        total_pixels += numpy_data.shape[0] * numpy_data.shape[1]
    
    # Calculate mean and std
    mean_rgb = sum_rgb / total_pixels
    std_rgb = np.sqrt((sum_squared_rgb / total_pixels) - (mean_rgb ** 2))
    
    #sum_rgb = np.array([0.0, 0.0, 0.0])
    #sum_squared_rgb = np.array([0.0, 0.0, 0.0])
    #total_pixels = 0
    
    #for data, _ in ds_train:
        #numpy_data = np.array(data) / 255.0  # Assumes data is a PIL Image
       # sum_rgb += numpy_data.sum(axis=(0, 1))
        #sum_squared_rgb += (numpy_data ** 2).sum(axis=(0, 1))
        #total_pixels += numpy_data.shape[0] * numpy_data.shape[1]
    
    #mean_rgb = sum_rgb / total_pixels
    #std_rgb = np.sqrt((sum_squared_rgb / total_pixels) - (mean_rgb ** 2))
    
    # Ensure the return format is explicitly a tuple of tuples
    #return tuple(mean_rgb), tuple(std_rgb)
    
    return mean_rgb, std_rgb
