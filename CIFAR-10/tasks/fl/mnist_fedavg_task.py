from cProfile import label
import random
from collections import defaultdict
from typing import List, Any, Dict
import logging
from copy import deepcopy
import math

import numpy as np
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split
from torchvision.utils import save_image
import torch
import torchvision
from torchvision import transforms
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from tasks.fl.fl_task import FederatedLearningTask
from models.simple import SimpleNet
from metrics.metric import Metric
from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from tasks.fl.fl_user import FLUser
from tasks.batch import Batch
from utils.parameters import Params

logger = logging.getLogger("logger")


class MNIST_FedAvgTask(FederatedLearningTask):
    params: Params = None

    models: Module = None

    criterion: Module = None
    scheduler: CosineAnnealingLR = None
    metrics: List[Metric] = None

    "Generic normalization for input data."
    input_shape: torch.Size = None

    fl_train_loaders: List[Any] = None
    fl_test_loaders: List[Any] = None
    ignored_weights = ["num_batches_tracked"]  # ['tracked', 'running']
    adversaries: List[int] = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction="mean")

    def make_scheduler(self) -> None:
        if self.params.scheduler:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.params.epochs)

    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    def accumulate_metrics(self, metrics, outputs, labels):
        for metric in metrics:
            metric.accumulate_on_batch(outputs, labels)

    def reset_metrics(self, metrics):
        for metric in metrics:
            metric.reset_metric()

    def report_metrics(
        self, metrics, step, prefix="", tb_writer=None, tb_prefix="Metric/"
    ):
        metric_text = []
        for metric in metrics:
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        logger.warning(f"{prefix} {step:4d}. {' | '.join(metric_text)}")

        return metrics[0].get_main_metric_value(), metrics[1].get_main_metric_value()

    def get_metrics(self, metrics):
        return metrics[0].get_main_metric_value(), metrics[1].get_main_metric_value()

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.local_model = self.build_model()
        self.criterion = self.make_criterion()
        self.adversaries = self.sample_adversaries()

        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()
        return

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        sampled_ids = random.sample(
            range(self.params.fl_total_participants), self.params.fl_no_models
        )

        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            test_loader = self.fl_test_loaders[user_id]
            compromised = self.check_user_compromised(user_id)
            user = FLUser(
                user_id,
                compromised=compromised,
                train_loader=train_loader,
                test_loader=test_loader,
            )
            sampled_users.append(user)

        return sampled_users

    def check_user_compromised(self, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = user_id in self.adversaries

        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f"Running vanilla FL, no attack.")
        elif self.params.fl_single_epoch_attack is None:
            adversaries_ids = random.sample(
                range(self.params.fl_total_participants),
                self.params.fl_number_of_adversaries,
            )
            logger.warning(
                f"Attacking over multiple epochs with following "
                f"users compromised: {adversaries_ids}."
            )
        else:
            logger.warning(
                f"Attack only on epoch: "
                f"{self.params.fl_single_epoch_attack} with "
                f"{self.params.fl_number_of_adversaries} compromised"
                f" users."
            )

        return adversaries_ids

    def get_fl_update(self, local_models, global_models) -> Dict[str, torch.Tensor]:
        local_updates = []
        for i in range(len(global_models)):
            global_model = global_models[i]
            local_model = local_models[i]

            local_update = dict()
            for name, data in local_model.state_dict().items():
                if self.check_ignored_weights(name):
                    continue
                local_update[name] = data - global_model.state_dict()[name]

            local_updates.append(local_update)

        return local_updates

    def get_avg_model(self, weight_accumulator):
        model = SimpleNet(num_classes=len(self.classes)).to(self.params.device)
        state_dict = {}
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = 1 / self.params.fl_total_participants
            average_update = scale * sum_update
            state_dict[name] = average_update
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_avg_model_weighted(self, weight_accumulator, total_weight):
        model = SimpleNet(num_classes=len(self.classes)).to(self.params.device)
        state_dict = {}
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = 1 / total_weight
            average_update = scale * sum_update
            state_dict[name] = average_update
        model.load_state_dict(state_dict, strict=False)
        return model

    def dp_clip(self, local_update_tensor: torch.Tensor, update_norm):
        if self.params.fl_dp_clip is not None and update_norm > self.params.fl_dp_clip:
            norm_scale = self.params.fl_dp_clip / update_norm
            local_update_tensor.mul_(norm_scale)

    def dp_add_noise(self, sum_update_tensor: torch.Tensor):
        if self.params.fl_dp_noise is not None:
            noised_layer = torch.FloatTensor(sum_update_tensor.shape)
            noised_layer = noised_layer.to(self.params.device)
            noised_layer.normal_(mean=0, std=self.params.fl_dp_noise)
            sum_update_tensor.add_(noised_layer)

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if self.check_ignored_weights(name):
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False

    def build_model(self):
        model = SimpleNet(num_classes=len(self.classes)).to(self.params.device)
        return model

    def load_data(self) -> None:
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        train_loaders, test_loaders = self.assign_data(bias=self.params.fl_q)
        self.fl_train_loaders = train_loaders
        self.fl_test_loaders = test_loaders
        return

    def assign_data(self, bias=1, p=0.1):
        num_labels = len(self.classes)
        num_workers = self.params.fl_total_participants
        server_pc = 0

        # assign data to the clients
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        # assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        server_data = []
        server_label = []

        # compute the labels needed for each class
        real_dis = [1.0 / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.0
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[: num_labels - 1])

        # randomly assign the data points based on the labels
        server_counter = [0 for _ in range(num_labels)]
        for x, y in self.train_dataset:
            upper_bound = y * (1.0 - bias) / (num_labels - 1) + bias
            lower_bound = y * (1.0 - bias) / (num_labels - 1)
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(
                    np.floor((rd - upper_bound) / other_group_size) + y + 1
                )
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y

            if server_counter[y] < samp_dis[y]:
                server_data.append(x)
                server_label.append(y)
                server_counter[y] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(
                    worker_group * worker_per_group
                    + int(np.floor(rd * worker_per_group))
                )
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

        random_order = np.random.RandomState(seed=self.params.random_seed).permutation(
            num_workers
        )
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        train_loaders, test_loaders = [], []
        transform_list = [
            transforms.RandomRotation((degree, degree))
            for degree in self.params.fl_client_degrees
        ]
        for i in range(len(each_worker_data)):
            train_set = ClientDataset(
                each_worker_data[i], each_worker_label[i], transform_list[i]
            )
            if self.params.fl_client_data is not None:
                tot = self.params.fl_client_data[i]
                train_size = int(tot * self.params.fl_client_train_ratio)
                test_size = tot - train_size
                train_set, test_set, _ = random_split(
                    train_set,
                    lengths=[
                        train_size,
                        test_size,
                        len(train_set) - train_size - test_size,
                    ],
                    generator=torch.Generator().manual_seed(self.params.random_seed),
                )
            else:
                tot = len(train_set)
                train_size = int(tot * self.params.fl_client_train_ratio)
                test_size = tot - train_size
                train_set, test_set = random_split(
                    train_set,
                    lengths=[train_size, test_size],
                    generator=torch.Generator().manual_seed(self.params.random_seed),
                )

            train_loader = DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                shuffle=False,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                shuffle=False,
                drop_last=True,
            )
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        return train_loaders, test_loaders


class ClientDataset(Dataset):
    def __init__(self, data_list, label_list, transform):
        super().__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return self.transform(self.data_list[index]), self.label_list[index]
