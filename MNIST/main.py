import argparse
import shutil
from datetime import datetime
import copy
from threading import Thread
import copy
import random
import itertools

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
cost = [random.uniform(0.004, 0.005) for _ in range(100)]
delta = 10
alpha = 0.384
beta = 0.4
s_delta = 10
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
    global_model = hlpr.task.model
    s_dict = {}
    for participant in range(hlpr.params.fl_total_participants):
        s_dict[participant] = 1.0
    logger.info(s_dict)
    
    # print(len(round_participants[0].test_loader))
    for epoch in range(hlpr.params.epochs + 1):
        round_participants = hlpr.task.sample_users_for_round()
    
        grads, grads1 = [], []
        grads_dict, grads1_dict = {}, {}
        accs = []
        
        logger.info(f"Epoch: {epoch}, sample {[agent.user_id for agent in round_participants]}")
        remaining_clients = len(round_participants)
        
        if hlpr.task.params.method != "br-shap":
            remaining_clients = len(round_participants)
            while remaining_clients > 0:
                thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
                threads = []
                for user in round_participants[len(round_participants) - remaining_clients: \
                                        len(round_participants) - remaining_clients + thread_pool_size]:
                    thread = ClientThread(user, hlpr, copy.deepcopy(global_model), user.user_id, s_dict, "Update", round_participants)
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    user_id, grad, acc, s = thread.join()
                    grads.append(grad)
                    accs.append(acc)
                    s_dict[user_id] = s
                remaining_clients -= thread_pool_size
        else:
            while remaining_clients > 0:
                thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
                threads = []
                for user in round_participants[len(round_participants) - remaining_clients: \
                                        len(round_participants) - remaining_clients + thread_pool_size]:
                    thread = ClientThread(user, hlpr, copy.deepcopy(global_model), user.user_id, s_dict, "Train", round_participants)
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    user_id, grad, grad1 = thread.join()
                    grads.append(grad)
                    grads_dict[user_id] = grad
                    grads1.append(grad1)
                    grads1_dict[user_id] = grad1
                remaining_clients -= thread_pool_size
            
            ## Update all the shares
            num_perms = 1
            sampled_agents = [agent.user_id for agent in round_participants]
            model_for_measure_share1 = copy.deepcopy(global_model)
            model_for_measure_share2 = copy.deepcopy(global_model)
            avg_shapley_share = {agent.user_id: 0 for agent in round_participants}
            for _ in range(num_perms):
                perm = sampled_agents
                random.shuffle(perm)
                shapley_share = {agent.user_id: 0 for agent in round_participants}

                for j in range(1, len(perm)+1):
                    agent_i = perm[j-1]
                    agents_with_i = perm[:j]
                    
                    ## Measure the sum of accuracies when using the first j datasets
                    new_state_dict = dict()
                    for name, _ in grads_dict[agent_i].items():
                        new_state_dict[name] = model_for_measure_share1.state_dict()[name]
                    for name in new_state_dict.keys():
                        for agent in agents_with_i:
                            new_state_dict[name].sub_(grads_dict[agent][name] * hlpr.params.lr)
                    model_for_measure_share1.load_state_dict(new_state_dict, strict=False)
                    remaining_clients = 1 if hlpr.params.idtest else len(round_participants)
                    accs = []
                    while remaining_clients > 0:
                        thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
                        threads = []
                        for user in round_participants[len(round_participants) - remaining_clients: \
                                                len(round_participants) - remaining_clients + thread_pool_size]:
                            thread = ClientThread(user, hlpr, copy.deepcopy(model_for_measure_share1), user.user_id, s_dict, "Test", round_participants)
                            threads.append(thread)
                            thread.start()
                        for thread in threads:
                            acc = thread.join()
                            accs.append(acc)
                        remaining_clients -= thread_pool_size

                    ## Measure the sum of accuracies when using the first j datasets 
                    #  with s_j improved by one
                    new_state_dict = dict()
                    for name, _ in grads_dict[agent_i].items():
                        new_state_dict[name] = model_for_measure_share2.state_dict()[name]
                    for name in new_state_dict.keys():
                        for agent in agents_with_i[:-1]:
                            new_state_dict[name].sub_(grads_dict[agent][name] * hlpr.params.lr)
                        new_state_dict[name].sub_(grads1_dict[agent_i][name] * hlpr.params.lr)
                    model_for_measure_share2.load_state_dict(new_state_dict, strict=False)
                    
                    remaining_clients = 1 if hlpr.params.idtest else len(round_participants)
                    acc1s = []
                    while remaining_clients > 0:
                        thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
                        threads = []
                        for user in round_participants[len(round_participants) - remaining_clients: \
                                                len(round_participants) - remaining_clients + thread_pool_size]:
                            thread = ClientThread(user, hlpr, copy.deepcopy(model_for_measure_share2), user.user_id, s_dict, "Test", round_participants)
                            threads.append(thread)
                            thread.start()
                        for thread in threads:
                            acc = thread.join()
                            acc1s.append(acc)
                        remaining_clients -= thread_pool_size

                    shapley_share[agent_i] = sum(acc1s) - sum(accs)

                for i in range(len(perm)):
                    agent_i = perm[i]
                    avg_shapley_share[agent_i] += shapley_share[agent_i]
            
            for agent_i in round_participants:
                avg_shapley_share[agent_i.user_id] = avg_shapley_share[agent_i.user_id] / num_perms
                if helper.params.idtest:
                    avg_shapley_share[agent_i.user_id] *= helper.params.fl_no_models
            
            logger.info(f"Epoch: {epoch}, shapley share: {avg_shapley_share}")

            for agent in round_participants:
                _s = s_dict[agent.user_id] + delta * (avg_shapley_share[agent.user_id] / s_delta - cost[agent.user_id])
                if _s < 0 or _s >= max_resource:
                    continue
                else:
                    s_dict[agent.user_id] = _s
        
        if epoch % 10 == 0:
            accs = []
            costs = [cost[i] * s_dict[i] for i in range(hlpr.params.fl_total_participants)]
            all_users = hlpr.task.all_users()
            remaining_clients = len(all_users)
            while remaining_clients > 0:
                thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
                threads = []
                for user in all_users[len(all_users) - remaining_clients: \
                                        len(all_users) - remaining_clients + thread_pool_size]:
                    thread = ClientThread(user, hlpr, copy.deepcopy(global_model), user.user_id, s_dict, "Test", round_participants)
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    acc = thread.join()
                    accs.append(acc)
                remaining_clients -= thread_pool_size

            logger.info(s_dict)
            logger.info(', '.join(map(str, accs)))
            logger.warning('Epoch: {} Sum of s_i: {}'.format(epoch, sum(s_dict.values())))
            logger.warning('Epoch: {}, Acc: {:.3f}, Sum of s_i: {}, Costs: {}'.format(epoch, sum(accs), sum(s_dict.values()), sum(costs)))
        
        ## Update the global model
        new_state_dict = dict()
        for name, _ in grads[0].items():
            new_state_dict[name] = global_model.state_dict()[name]
        for name in new_state_dict.keys():
            for grad in grads:
                new_state_dict[name].sub_(grad[name] * hlpr.params.lr)
        global_model.load_state_dict(new_state_dict, strict=False)
        
class ClientThread(Thread):
    def __init__(self, user, hlpr, global_model, _id, s_all, task, sampled_agents, _lr = 0.01):
        super().__init__()
        self.user = user
        self.hlpr = hlpr
        self.model = global_model
        self.id = _id
        self.s_all = s_all
        self.s = s_all[_id]
        self.sampled_agents = [agent.user_id for agent in sampled_agents]
        self.eps = 0.001
        self.task = task
        self.learning_rate = _lr
        self._return = None

    def cost(self, s):
        return self.cost_per_s * s
    
    def avg_cost_of_others(self):
        return self.cost_per_s * (sum(self.s_all.values()) - self.s) * 1.0 / (len(self.s_all) - 1) 
    
    def cost_gradient(self):
        return cost[self.id]
    
    def run(self):
        if self.task == "Train":
            # acc = self.test(self.model)
            criterion = torch.nn.CrossEntropyLoss()
            self.model.train()

            self.model.zero_grad()
            for i, data in enumerate(self.user.train_loader):
                if i + 1 > self.s:
                    break
                batch = self.hlpr.task.get_batch(i, data)
                logits = self.model(batch.inputs)
                loss = criterion(logits, batch.labels)
                loss.backward()
            
            grad, grad_eps = {}, {}
            if int(self.s) != 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        grad[name] = param.grad / int(self.s)
            else:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        grad[name] = torch.zeros_like(param)
            
            # Run for s_delat more rounds for approximating the gradient of shapley share
            for i, data in enumerate(self.user.train_loader, start=int(self.s)):
                if i + 1 > s_delta:
                    break
                batch = self.hlpr.task.get_batch(i, data)
                logits = self.model(batch.inputs)
                loss = criterion(logits, batch.labels)
                loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad_eps[name] = param.grad / (int(self.s) + s_delta)

            self._return = self.user.user_id, grad, grad_eps

        elif self.task == "Update":
            s_all = self.s_all
            fl = FLInstance(len(self.s_all), self.s_all, alpha, beta, 0.01)
            if self.hlpr.params.method == "br-shap":
                # FedBR-SV, ∂d/∂s = ∂φ/∂s - c
                # d = fl.compute_shapley_value_derivative(self.id) - self.cost_gradient()
                raise ValueError("Unsupported method!")
            elif self.hlpr.params.method == "br":
                # FedBR, ∂d/∂s = ∂a/∂s - c
                d = a_derivative(s_all.values()) - self.cost_gradient()    
            elif self.hlpr.params.method == "br-bg":
                # FedBR-BG, ∂d/∂s = ∂a/∂s - (1-β)*c
                d = a_derivative(s_all.values()) - (1 - cost_scalar_beta) * self.cost_gradient()
            else:
                s_all_add_delta_s = s_all.copy()
                s_all_add_delta_s[self.id] += d_s
                d = (a(s_all_add_delta_s.values()) - a(s_all.values())) / d_s - self.cost_gradient()
            s_ = self.s + delta * d
            if not (s_ > max_resource or s_ < 0):
                self.s = s_
            self._return = self.user.user_id, self.s
        
        elif self.task == "Test":
            self._return = self.test(self.model)
        else:
            raise ValueError("Task not recognized!")
        
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
    parser.add_argument('--idtest', dest='idtest', action='store_true', default=False, help='whether to use identical testing data')
    
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name
    params['method'] = args.method
    params['idtest'] = args.idtest
    helper = Helper(params)
    
    ## Make all the testing dataset the same
    if args.idtest:
        logger.info("All the testing data are made identical!")
        helper.task.merge_test_data()
    
    try:
        fl_run(helper)
    except (KeyboardInterrupt):
        logger.error(f"Fine. Deleted: {helper.params.folder_path}")
        shutil.rmtree(helper.params.folder_path, ignore_errors=True)
        if helper.params.tb:
            shutil.rmtree(f'runs/{args.name}')