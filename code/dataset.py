import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImbalanceCIFAR(Dataset):
    def __init__(self, cifar_version=10, root='./data', train=True, transform=None, imbalance_ratio=0.01,
                 dataset_Path=None, debug=False):
        super(ImbalanceCIFAR, self).__init__()
        self.debug = debug
        self.transform = transform

        # print('Creating the dataset.  The dataset path is', dataset_Path)
        if dataset_Path is None:
            if cifar_version == 10:
                print('Cifar version is 10')
                self.original_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                                                     transform=transform)
            elif cifar_version == 100:
                self.original_dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True,
                                                                      transform=transform)
            else:
                raise ValueError("CIFAR version must be 10 or 100")

            self.num_classes = 10 if cifar_version == 10 else 100
            self._create_long_tailed(imbalance_ratio)
        else:
            print('The dataset path is not none')
            directory = dataset_Path  # Set the directory you want to count
            print('Directory within dataset.py', directory)
            entries = os.listdir(directory)  # List all the entries in the directory
            # Count the number of directories
            folder_count = sum(os.path.isdir(os.path.join(directory, entry)) for entry in entries)
            self.num_classes = folder_count
            shuffle = True  # if train else False
            self.original_dataset = torchvision.datasets.ImageFolder(dataset_Path, transform)
            self._create_long_tailed(imbalance_ratio)

    def _create_long_tailed(self, imbalance_ratio):
        # Get class distribution
        class_counts = np.bincount([label for _, label in self.original_dataset])

        # Compute number of samples for each class with exponential decrease
        max_count = max(class_counts)

        num_samples_lt = [int(max_count * (imbalance_ratio ** (i / (self.num_classes - 1.0)))) for i in
                          range(self.num_classes)]
        self.indices = []
        self.targets = []
        for i in range(self.num_classes):
            class_indices = np.where(np.array(self.original_dataset.targets) == i)[0]
            np.random.shuffle(class_indices)
            selected_indices = class_indices[:num_samples_lt[i]]
            self.indices.extend(selected_indices)
            self.targets.extend([i] * len(selected_indices))
        np.random.shuffle(self.indices)
        if self.debug:
            new_len = 256
            self.targets = self.targets[:new_len]
            self.indices = self.indices[:new_len]
        print('The length of the dataset is', len(self.indices))

    def debug_mode(self):
        new_len = 256
        self.targets = self.targets[:new_len]
        self.indices = self.indices[:new_len]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.original_dataset[real_idx]

    def get_class_distribution(self):
        class_counts = np.bincount([self.original_dataset.targets[idx] for idx in self.indices])
        return {i: class_counts[i] for i in range(self.num_classes)}

    def get_class_names(self):
        return self.original_dataset.classes

    def plot_class_distribution(self, dataset="bird", path=f'./class_distribution.png'):
        distribution = self.get_class_distribution()
        # Sort classes by the number of samples per class
        sorted_classes = sorted(distribution.items(), key=lambda item: item[1], reverse=True)

        # Separate the class indices and their corresponding counts
        sorted_indices, sorted_counts = zip(*sorted_classes)

        # Determine the threshold for minority classes, for example, you might define it as the lower 20%
        threshold = np.percentile(sorted_counts, 30)

        # Create a line plot for class distribution
        plt.plot(sorted_indices, sorted_counts, label='Class Distribution')

        # Fill the area under the curve
        plt.fill_between(sorted_indices, sorted_counts, where=(np.array(sorted_counts) <= threshold), color='red',
                         alpha=0.5, label='Minority classes')
        plt.fill_between(sorted_indices, sorted_counts, where=(np.array(sorted_counts) > threshold), color='green',
                         alpha=0.5, label='Majority classes')

        # Add labels and title
        plt.xlabel('Sorted class indices (Large → Small)')
        plt.ylabel('Training samples per class')
        plt.title('Class Distribution in Dataset')
        plt.legend()

        # Show the plot
        plt.savefig(path)
        plt.close()

    def plot_class_distribution_imbalanced(self):
        # Count the occurrences of each label
        labels = self.targets
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Extract labels and counts for plotting
        unique_labels = list(label_counts.keys())
        count_values = list(label_counts.values())

        # Create a histogram
        plt.bar(unique_labels, count_values)
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        plt.title("Label Distribution Histogram")
        plt.show()

    def get_samples_from_each_class(self, num_samples=1):
        samples = {}
        for i in range(self.num_classes):
            class_indices = [idx for idx in self.indices if self.original_dataset.targets[idx] == i]
            np.random.shuffle(class_indices)
            samples[i] = [self.original_dataset[class_indices[j]] for j in range(num_samples)]
        return samples

    def imshow(img):
        img = img.numpy().transpose((1, 2, 0))  # Convert from tensor image
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = std * img + mean  # Unnormalize
        img = np.clip(img, 0, 1)  # Clip to [0, 1]
        plt.imshow(img)
        plt.show()

    def show_augmented_images(self, image_index, augmentations, num_samples=5):
        original_image, _ = self.original_dataset[image_index]
        images = [augmentations(original_image) for _ in range(num_samples)]
        grid_image = torchvision.utils.make_grid(images, nrow=num_samples)
        self.imshow(grid_image)

    def compute_imbalance_ratio(self):
        class_distribution = self.get_class_distribution()
        max_count = max(class_distribution.values())
        min_count = min(class_distribution.values())
        return max_count / min_count

    def get_random_batch(self, batch_size=32):
        indices = np.random.choice(self.indices, batch_size, replace=False)
        return [self.original_dataset[idx] for idx in indices]

    def extract_features_and_labels(self, dataset):
        features = []
        labels = []

        # Assuming the dataset yields (image, label) tuples where image is a PyTorch tensor
        for img, label in dataset:
            # Flatten the image and convert to a NumPy array or a single vector tensor
            flattened_img = img.view(-1).numpy()  # If the image is a PyTorch tensor
            features.append(flattened_img)
            labels.append(label)

        # Convert lists to NumPy arrays
        return np.array(features), np.array(labels)

    def visualize_with_tsne(self, features, labels):
        # Select a subset of data for visualization to avoid overplotting
        # For instance, randomly select 1000 samples if the dataset is large
        if len(labels) > 1000:
            indices = np.random.choice(range(len(labels)), 1000, replace=False)
            features_subset = features[indices]
            labels_subset = labels[indices]
        else:
            features_subset = features
            labels_subset = labels

        tsne = TSNE(n_components=2, random_state=123)
        tsne_results = tsne.fit_transform(features_subset)

        # Set up markers and colors
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H']
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_classes))

        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            indices = labels_subset == i
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        alpha=0.6, marker=markers[i % len(markers)],
                        c=[colors[i]], label=self.original_dataset.classes[i])

        plt.title('t-SNE visualization of the dataset')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.savefig('./tsne.png')

    def visualize_with_pca(self, features, labels):
        # Select a subset of data for visualization to avoid overplotting
        if len(labels) > 1000:
            indices = np.random.choice(range(len(labels)), 1000, replace=False)
            features_subset = features[indices]
            labels_subset = labels[indices]
        else:
            features_subset = features
            labels_subset = labels

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(features_subset)

        # Set up markers and colors
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H']
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_classes))

        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            indices = labels_subset == i
            plt.scatter(pca_results[indices, 0], pca_results[indices, 1],
                        alpha=0.6, marker=markers[i % len(markers)],
                        c=[colors[i]], label=self.original_dataset.classes[i])
        plt.legend()
        plt.title('PCA visualization of the dataset')
        plt.xlabel('PCA feature 1')
        plt.ylabel('PCA feature 2')
        plt.savefig('./pca.png')


class ImbalanceCIFAR10(ImbalanceCIFAR):
    def __init__(self, root='./data', train=True, transform=None, imbalance_ratio=0.1, debug=0):
        super(ImbalanceCIFAR10, self).__init__(cifar_version=10, root=root, train=train, transform=transform,
                                               imbalance_ratio=imbalance_ratio, debug=debug)


class ImbalanceCIFAR100(ImbalanceCIFAR):
    def __init__(self, root='./data', train=True, transform=None, imbalance_ratio=0.1, debug=0):
        super(ImbalanceCIFAR100, self).__init__(cifar_version=100, root=root, train=train, transform=transform,
                                                imbalance_ratio=imbalance_ratio, debug=debug)
