class MixupLTDataloader():
    def __init__(self, batch_size, dataset, alpha=0.2):
        self.n_samples = len(dataset)
        self.n_classes = dataset.num_classes
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_iterations = math.ceil(self.n_samples / batch_size)
        self.labels = np.array(dataset.targets)
        self.dataset = dataset

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
        print(type(rand_sample))
        return rand_sample , rand_label

    def generate_sample(self):
        rand_class = self.random_class()
        return self.random_sample(rand_class)

    def generate_mixup(self):
        # Define two samplers
        probability = torch.zeros(self.n_classes)

        x1, y1 = self.generate_sample()
        x2, y2 = self.generate_sample()

        x_hat = self.alpha * x1 + (1 - self.alpha) * x2
        y_hat = self.alpha * y1 + (1 - self.alpha) * y2

        c1 = math.floor(y_hat)
        decimal = y_hat - c1
        if c1 != self.n_classes -1:
        c2 = c1 + 1
        probability[c1] = 1 - decimal
        probability[c2] = decimal
        return x_hat , torch.FloatTensor(probability)

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    raise NotImplemented('You must do.')
