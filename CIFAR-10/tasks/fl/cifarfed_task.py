import random
from collections import defaultdict

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import functional as F
from tasks.cifar10_task import Cifar10Task
from tasks.fl.fl_task import FederatedLearningTask


class CifarFedTask(FederatedLearningTask, Cifar10Task):
    def load_data(self) -> None:
        self.load_cifar_data()
        print(f"Rotation Angles: {self.params.rotation_angles}")
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants, alpha=self.params.fl_dirichlet_alpha
            )
            train_loaders = [
                (pos, self.get_train(indices))
                for pos, indices in indices_per_participant.items()
            ]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            all_train_range = list(range(len(self.train_dataset)))
            all_test_range = list(range(len(self.test_dataset)))
            random.shuffle(all_train_range)
            random.shuffle(all_test_range)
            train_loaders = [
                self.get_train_old(all_train_range, pos, self.params.rotation_angles)
                for pos in range(self.params.fl_total_participants)
            ]
            test_loaders = [
                self.get_test_old(all_test_range, pos, self.params.rotation_angles)
                for pos in range(self.params.fl_total_participants)
            ]
        print("Sum of train ", sum([len(loader) for loader in train_loaders]))
        print("Sum of test ", sum([len(loader) for loader in test_loaders]))
        self.fl_train_loaders = train_loaders
        self.fl_test_loaders = test_loaders
        return

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indices dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as
        parameters for
        dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if (
                ind in self.params.poison_images
                or ind in self.params.poison_images_test
            ):
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha])
            )
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][: min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][
                    min(len(cifar_classes[n]), no_imgs) :
                ]

        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param indices:
        :return:
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            sampler=SubsetRandomSampler(indices),
        )
        return train_loader

    def get_train_old(self, all_range, model_no, rotation_angles):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len : (model_no + 1) * data_len]
        # train_loader = DataLoader(self.train_dataset,
        #                           batch_size=self.params.batch_size,
        #                           sampler=SubsetRandomSampler(
        #                               sub_indices))

        if rotation_angles:
            subset_dataset = ClientDataSet(
                self.train_dataset, sub_indices, rotation_angles[model_no]
            )
        else:
            subset_dataset = ClientDataSet(self.train_dataset, sub_indices)
        train_loader = DataLoader(
            subset_dataset, batch_size=self.params.batch_size, shuffle=True
        )

        return train_loader

    def get_test_old(self, all_range, model_no, rotation_angles):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.test_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len : (model_no + 1) * data_len]
        # test_loader = DataLoader(self.test_dataset,
        #                           batch_size=self.params.batch_size,
        #                           sampler=SubsetRandomSampler(
        #                               sub_indices))

        if rotation_angles:
            subset_dataset = ClientDataSet(
                self.train_dataset, sub_indices, rotation_angles[model_no]
            )
        else:
            subset_dataset = ClientDataSet(self.train_dataset, sub_indices)

        test_loader = DataLoader(
            subset_dataset, batch_size=self.params.batch_size, shuffle=False
        )

        return test_loader


class ClientDataSet(Dataset):
    def __init__(self, dataset, indices, rotate_angle=0):
        self.dataset = dataset
        self.indices = indices
        self.rotate_angle = rotate_angle
        self.transformed_images = []

        for idx in self.indices:
            image, label = self.dataset[idx]
            image = F.rotate(image, self.rotate_angle)
            self.transformed_images.append((image, label))

    def __getitem__(self, index):
        return self.transformed_images[index]

    def __len__(self):
        return len(self.indices)
