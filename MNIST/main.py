import argparse
import shutil
from datetime import datetime
import copy
from threading import Thread
import copy
import random

import yaml
from prompt_toolkit import prompt
from shapley_value import FLInstance
from helper import Helper
from utils.utils import *

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger('logger')

## MNIST
max_resource = 100
d_s = 5
cost = [random.uniform(0.004, 0.005) for _ in range(10)]
delta = 1
alpha = 0.384
beta = 0.4
cost_scalar_beta = 0.8

def a(s_all):
    return 1 - (alpha * 1.0) / pow(sum(s_all), beta)

def a_derivative(s_all):
    return (alpha * beta * 1.0) / pow(sum(s_all), beta + 1)

def round_to_nearest_factor(num, factor):
    if factor == 0:
        raise ValueError("Factor must be non-zero.")
    nearest_lower_multiple = (num // factor) * factor
    remainder = num % factor
    if remainder >= factor / 2:
        return nearest_lower_multiple + factor
    else:
        return nearest_lower_multiple


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=False, ratio=None, report=False):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, ratio)
        loss.backward()
        optimizer.step()

    return


def test(hlpr: Helper, model, test_loader):
    model.eval()

    metrics = copy.deepcopy(hlpr.task.metrics)
    hlpr.task.reset_metrics(metrics)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            batch = hlpr.task.get_batch(i, data)
            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(metrics, outputs=outputs, labels=batch.labels)
        test_acc, test_loss = hlpr.task.get_metrics(metrics)

    return test_acc, test_loss


def fl_run(hlpr: Helper):
    hlpr.task.model = hlpr.task.build_model()
    round_participants = hlpr.task.sample_users_for_round()
    s_dict = {}
    for participant in round_participants:
        s_dict[participant.user_id] = 1.0
    logger.info(s_dict)
    
    print(len(round_participants[0].test_loader))
    for epoch in range(hlpr.params.epochs + 1):
        global_model = hlpr.task.model
        grads, accs = [], []
        
        remaining_clients = len(round_participants)
        while remaining_clients > 0:
            thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
            threads = []
            for user in round_participants[len(round_participants) - remaining_clients: \
                                    len(round_participants) - remaining_clients + thread_pool_size]:
                thread = ClientThread(user, hlpr, copy.deepcopy(global_model), user.user_id, s_dict)
                threads.append(thread)
                thread.start()
            for thread in threads:
                user_id, grad, acc, s = thread.join()
                grads.append(grad)
                accs.append(acc)
                s_dict[user_id] = s
            remaining_clients -= thread_pool_size
        
        logger.info(s_dict)
        logger.info(', '.join(map(str, accs)))

        new_state_dict = dict()
        for name, _ in grads[0].items():
            new_state_dict[name] = global_model.state_dict()[name]
        for name in new_state_dict.keys():
            for grad in grads:
                new_state_dict[name].sub_(grad[name] * hlpr.params.lr)

        global_model.load_state_dict(new_state_dict, strict=False)

        logger.warning('Epoch: {}, Sum of Accs: {:.3f}, Sum of s_i: {}'.format(epoch, sum(accs), sum(s_dict.values())))


class ClientThread(Thread):
    def __init__(self, user, hlpr, global_model, id, s_all):
        super().__init__()
        self.user = user
        self.hlpr = hlpr
        self.model = global_model
        self.id = id
        self.s_all = s_all
        self.s = s_all[id]
        self._return = None

    def cost(self, s):
        return self.cost_per_s * s
    
    def avg_cost_of_others(self):
        return self.cost_per_s * (sum(self.s_all.values()) - self.s) * 1.0 / (len(self.s_all) - 1) 
    
    def cost_gradient(self):
        return cost[self.id]
    
    def utility(self, s_all, s):
        return a(s_all) - self.cost(s)
    
    def run(self):
        fl = FLInstance(len(self.s_all), self.s_all, alpha, beta, 0.01)
        acc = self.test(self.model)
        s = self.s
        s_all = self.s_all
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        for i, data in enumerate(self.user.train_loader):
            batch = self.hlpr.task.get_batch(i, data)
            logits = self.model(batch.inputs)
            loss = criterion(logits, batch.labels)
            loss.backward()
            if i >= s:
                break
        
        grad = {}
        if s != 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad[name] = param.grad / self.s
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad[name] = torch.zeros_like(param)
        
        if self.hlpr.params.method == "br-shap":
            # FedBR-SV, ∂d/∂s = ∂φ/∂s - c
            d = fl.compute_shapley_value_derivative(self.id) - self.cost_gradient()
        elif self.hlpr.params.method == "br":
            # FedBR, ∂d/∂s = ∂a/∂s - c
            d = a_derivative(s_all.values()) - self.cost_gradient()    
        elif self.hlpr.params.method == "br-bg":
            # FedBR-BG, ∂d/∂s = ∂a/∂s - (1-β)*c
            d = a_derivative(s_all.values()) - (1 - cost_scalar_beta) * self.cost_gradient()
            # print(d)   
        else:
            s_all_add_delta_s = s_all.copy()
            s_all_add_delta_s[self.id] += d_s
            d = (a(s_all_add_delta_s.values()) - a(s_all.values())) / d_s - self.cost_gradient()
        s_ = s + delta * d
        if not (s_ > max_resource or s_ < 0):
            s = s_
        self._return = self.user.user_id, grad, acc, s

    def test(self, model):
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.user.test_loader):
                batch = self.hlpr.task.get_batch(i, data)
                logits = model(batch.inputs)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                correct += int((preds == batch.labels).sum())
                total += batch.labels.size(0)
        acc = correct / total

        return acc

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='fedavg.yaml')
    parser.add_argument('--name', dest='name', default='test', help='Tensorboard name')
    parser.add_argument('--method', dest='method', default='br-shap', help='Tensorboard name')
    
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name
    params['method'] = args.method
    helper = Helper(params)

    if args.method != "br-shap":
        delta = 3    
    try:
        fl_run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
