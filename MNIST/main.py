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
        
        print([agent.user_id for agent in round_participants])
        remaining_clients = len(round_participants)
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
        print(avg_shapley_share)

        for agent in round_participants:
            _s = s_dict[agent.user_id] + delta * (avg_shapley_share[agent.user_id] / s_delta - cost[agent.user_id])
            if _s < 0 or _s >= max_resource:
                continue
            else:
                s_dict[agent.user_id] = _s
        
        accs = []
        if epoch % 10 == 0:
            accs = []
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

        ## Update the global model
        new_state_dict = dict()
        for name, _ in grads[0].items():
            new_state_dict[name] = global_model.state_dict()[name]
        for name in new_state_dict.keys():
            for grad in grads:
                new_state_dict[name].sub_(grad[name] * hlpr.params.lr)
        global_model.load_state_dict(new_state_dict, strict=False)

        logger.info(s_dict)
        logger.info(', '.join(map(str, accs)))
        logger.warning('Epoch: {} Sum of s_i: {}'.format(epoch, sum(s_dict.values())))
        logger.warning('Epoch: {}, Sum of Accs: {:.3f}, Len of Accs: {}, Sum of s_i: {}'.format(epoch, sum(accs), len(accs), sum(s_dict.values())))

class ClientThread(Thread):
    def __init__(self, user, hlpr, global_model, _id, s_all, task, sampled_agents, grads=None, grads1=None, _lr = 0.01):
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
        self.grads = grads
        self.grads1 = grads1
        self.learning_rate = _lr
        self._return = None

    def cost(self, s):
        return self.cost_per_s * s
    
    def avg_cost_of_others(self):
        return self.cost_per_s * (sum(self.s_all.values()) - self.s) * 1.0 / (len(self.s_all) - 1) 
    
    def cost_gradient(self):
        return cost[self.id]
    
    def utility(self, s_all, s):
        return a(s_all) - self.cost(s)


    # def update_share_by_real_contribution(self):
    #     n = len(self.sampled_agents)
    #     # num_perms = int(n * np.log(n) / self.eps) # sample n * log(n) / epsilon permutations
    #     num_perms = 1
    #     shapley_shares = dict()
        
    #     for agent_i in self.sampled_agents:
    #         shapley_share_of_agent_i = []
    #         for iter_idx in range(num_perms):
    #             perm = self.sampled_agents
    #             random.shuffle(perm)
    #             agent_i_idx = perm.index(agent_i)
    #             agents_before_i = perm[:agent_i_idx]
    #             global_model_with_updated_share = copy.deepcopy(self.model)

    #             new_state_dict = dict()
    #             for name, _ in self.grads[agent_i].items():
    #                 new_state_dict[name] = self.model.state_dict()[name]
    #             for name in new_state_dict.keys():
    #                 for agent in agents_before_i:
    #                     new_state_dict[name].sub_(self.grads[agent][name] * self.learning_rate)
    #             self.model.load_state_dict(new_state_dict, strict=False)
    #         #      while remaining_clients > 0:
    #         # thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
    #         # threads = []
    #         # for user in round_participants[len(round_participants) - remaining_clients: \
    #         #                         len(round_participants) - remaining_clients + thread_pool_size]:
                
    #         #     thread = ClientThread(user, hlpr, copy.deepcopy(global_model), user.user_id, s_dict, "Update", \
    #         #                           round_participants, grads_dict, grads1_dict, hlpr.params.lr)
    #         #     threads.append(thread)  
    #         #     thread.start()
    #         # for thread in threads:
    #         #     user_id, s = thread.join()
    #         #     s_dict[user_id] = s
    #         # remaining_clients -= thread_pool_size

    #         #     acc = self.test(self.model)
                
    #             new_state_dict = dict()
    #             for name, _ in self.grads[agent_i].items():
    #                 new_state_dict[name] = self.model.state_dict()[name]
    #             for name in new_state_dict.keys():
    #                 for agent in agents_before_i:
    #                     new_state_dict[name].sub_(self.grads[agent][name] * self.learning_rate)
    #                 new_state_dict[name].sub_(self.grads1[agent_i][name] * self.learning_rate)
    #             global_model_with_updated_share.load_state_dict(new_state_dict, strict=False)
    #             acc1 = self.test(global_model_with_updated_share)
    #             print(acc1, acc)
    #             shapley_share_of_agent_i.append(acc1 - acc)

    #         shapley_shares[agent_i] = sum(shapley_share_of_agent_i) / num_perms
    #     print(shapley_shares)
    
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
            grad = {}
            if int(self.s) != 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        grad[name] = param.grad / self.s
            else:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        grad[name] = torch.zeros_like(param)
            
            # One more round for computing the gradient of shapley share
            grad1 = {}
            self.model.zero_grad()
            for i, data in enumerate(self.user.train_loader):
                if i + 1 > self.s + s_delta:
                    break
                batch = self.hlpr.task.get_batch(i, data)
                logits = self.model(batch.inputs)
                loss = criterion(logits, batch.labels)
                loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad1[name] = param.grad / (self.s + s_delta)
            # acc1 = self.test(self.model)

            # if acc != acc1:
            #     print("Before:", acc, " After:", self.test(self.model))

            self._return = self.user.user_id, grad, grad1

        elif self.task == "Update":
            s_all = self.s_all
            # self.update_share_by_real_contribution()
            fl = FLInstance(len(self.s_all), self.s_all, alpha, beta, 0.01)
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
            s_ = self.s + delta * d
            if not (s_ > max_resource or s_ < 0):
                self.s = s_
            self._return = self.user.user_id, self.s
        
        elif self.task == "Test":
            self._return = self.test(self.model)

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
    if args.idtest:
        print("id test")
        helper.task.merge_test_data()

    # if args.method != "br-shap":
    #     delta = 10    
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
