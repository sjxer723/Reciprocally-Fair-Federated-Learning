import argparse
from datetime import datetime
import copy
from threading import Thread
import copy
import json
import random
from tqdm import tqdm
from itertools import product

import yaml
from shapley_value import FLInstance
from helper import Helper
from utils.utils import *
from scipy.optimize import curve_fit

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger("logger")

## MNIST
max_resource = 100
d_s = 5
costs = []
delta = 10
s_lr = 100
s_delta = 10
cost_scalar_beta = 0.8


def train(
    hlpr: Helper,
    epoch,
    model,
    optimizer,
    train_loader
):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        logits = model(batch.inputs)
        loss = criterion(logits, batch.labels)
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

class ClientThread(Thread):
    def __init__(self, user, hlpr, global_model, s_vec, task, sampled_agents, _lr=0.01):
        super().__init__()
        self.user = user
        self.hlpr = hlpr
        self.model = global_model
        self.id = user.user_id
        self.s_vec = s_vec
        self.s = s_vec[user.user_id]
        self.sampled_agents = [agent.user_id for agent in sampled_agents]
        self.eps = 0.001
        self.task = task
        self.learning_rate = _lr
        self._return = None

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
        elif self.task == "Test":
            self._return = self.id, self.test(self.model)
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


def fl_run_with_fixed_share(hlpr: Helper, s_vec, verbose=False):
    hlpr.task.model = hlpr.task.build_model()
    accs = [0.0 for _ in range(len(s_vec))]

    for epoch in range(hlpr.params.epochs + 1):
        global_model = hlpr.task.model
        grads = []
        round_participants = hlpr.task.sample_users_for_round()
        remaining_clients = len(round_participants)

        while remaining_clients > 0:
            thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
            threads = []
            for user in round_participants[
                len(round_participants) - remaining_clients : len(round_participants)
                - remaining_clients
                + thread_pool_size
            ]:
                thread = ClientThread(
                    user,
                    hlpr,
                    copy.deepcopy(global_model),
                    s_vec,
                    "Train",
                    round_participants,
                )
                threads.append(thread)
                thread.start()
            for thread in threads:
                _, grad, _ = thread.join()
                grads.append(grad)

            threads = []
            for user in round_participants[
                len(round_participants) - remaining_clients : len(round_participants)
                - remaining_clients
                + thread_pool_size
            ]:
                thread = ClientThread(
                    user,
                    hlpr,
                    copy.deepcopy(global_model),
                    s_vec,
                    "Test",
                    round_participants,
                )
                threads.append(thread)
                thread.start()
            for thread in threads:
                user_id, acc = thread.join()
                accs[user_id] = acc

            remaining_clients -= thread_pool_size

        new_state_dict = dict()
        for name, _ in grads[0].items():
            new_state_dict[name] = global_model.state_dict()[name]
        for name in new_state_dict.keys():
            for grad in grads:
                new_state_dict[name].sub_(grad[name] * hlpr.params.lr)

        global_model.load_state_dict(new_state_dict, strict=False)

        if verbose and epoch % 10 == 0:
            accs_str = ", ".join(["{:.2f}".format(acc) for acc in accs])
            logger.info("Epoch: {}, Accs: [{}]".format(epoch, accs_str))
    return accs


def best_response(hlpr: Helper, W, costs, verbose=False):
    hlpr.task.model = hlpr.task.build_model()
    all_users = hlpr.task.all_users()
    num_of_users = len(all_users)
    num_of_groups = len(W)
    s_vec = [1.0 for _ in range(num_of_users)]  # initial data share vector

    def derivative_of_accuracy(S, *W, j, grouping_fun=None):
        # W is a (num_of_groups * num_of_groups) matrix
        if grouping_fun is None:
            grouping_fun = lambda _: 0
        group_of_j = grouping_fun(j)
        w = W[group_of_j]
        len_of_S = len(S)
        return (
            w[group_of_j]
            / (1 + sum([w[grouping_fun(i)] * S[i] for i in range(len_of_S)])) ** 2
        )

    def partial_derivative_of_accuracy(S, *W, agents_with_i, i, grouping_fun=None):
        # Calculate the partial derivative of accuracy with respect to s_i
        if grouping_fun is None:
            grouping_fun = lambda _: 0
        len_of_S = len(S)
        derivative = 0
        for i in range(len_of_S):
            w = W[grouping_fun(i)]
            denominator = 1 + sum([w[grouping_fun(j)] * S[j] for j in agents_with_i])
            derivative += w[grouping_fun(i)] / (denominator**2)

        return derivative

    def update_share(S, W, i):
        derivative = derivative_of_accuracy(
            S, *W, j=i, grouping_fun=lambda x: int((x / num_of_users) * num_of_groups)
        )
        s_updated = S[i]
        if hlpr.params.method == "br":
            s_updated = S[i] + s_lr * (derivative - costs[i])
        elif hlpr.params.method == "br-bg":
            s_updated = S[i] + s_lr * (derivative - (1 - cost_scalar_beta) * costs[i])
        elif hlpr.params.method == "br-shap":
            # For FedBR-SV, we need to estimate of the Shapley value
            fl = FLInstance(num_of_users, S, alpha, beta, _eps=1)
            derivative_f = lambda agents_with_i, i: partial_derivative_of_accuracy(
                S,
                *W,
                agents_with_i=agents_with_i,
                i=i,
                grouping_fun=lambda x: int((x / num_of_users) * num_of_groups),
            )
            shapley_derivative = (
                fl.compute_derivative_of_shapley_value(i, derivative_f) - costs[i]
            )
            s_updated = S[i] + s_lr * shapley_derivative
        else:
            raise ValueError(
                "unknown method {} for best response".format(hlpr.params.method)
            )
        # print(s_updated)
        if s_updated >= max_resource or s_updated <= 0:
            return S[i]
        else:
            return s_updated

    for epoch in range(hlpr.params.num_of_br + 1):
        round_participants = hlpr.task.sample_users_for_round()
        ## Update share for every participant
        for user in round_participants:
            s_vec[user.user_id] = update_share(s_vec, W, user.user_id)
        ## Report the accuracy for all testing datasets
        if epoch % 10 == 0 and verbose:
            logger.info(s_vec)
            incurred_costs = [c * s for c, s in zip(costs, s_vec)]
            logger.warning(
                "Epoch: {}, Sum of s_i: {}, Costs: {}".format(
                    epoch, sum(s_vec), sum(incurred_costs)
                )
            )

    return s_vec


def non_iid_main(params: Params, rotation_angles=None, verbose=False):
    # partition the datasets into three parts
    fit_params = copy.deepcopy(params)
    types_of_data = 3
    fit_params["fl_total_participants"] = types_of_data
    fit_params["fl_no_models"] = types_of_data
    fit_params["epochs"] = 100
    fit_params["rotation_angles"] = rotation_angles  # for rotation
    fit_helper = Helper(fit_params)

    min_size_train_loader = min(
        200, min([len(user.train_loader) for user in fit_helper.task.all_users()])
    )
    s_step = 200
    num_of_samples_per_dimension = min_size_train_loader // s_step
    s_vecs = list(
        product([s_step * i for i in range(num_of_samples_per_dimension + 1)], repeat=3)
    )

    # a_i = 1 -  1 / (1 + w{i,1} * s1 + w{i, 2} * s2 + w{i, 3} * s3)
    def accuracy_func(S, *w):
        len_of_S = len(S)
        return 1 - 1 / (1 + sum([w[i] * S[i] for i in range(len_of_S)]))

    # Fit the accuracy function
    W = np.zeros((types_of_data, types_of_data))
    fl_results = {}
    if os.path.exists("out/{}_non_iid_fl_weights.json".format(params["task"])):
        past_results = json.load(
            open("out/{}_non_iid_fl_weights.json".format(params["task"]), "r")
        )
        W = np.array(past_results["W"])
        logger.info("Loaded previous fitted weights: {}".format(W))
    else:
        logger.info("Fitting the accuracy function...")
        all_accs = np.zeros((len(s_vecs), types_of_data))
        for s_idx, s_vec in tqdm(enumerate(s_vecs), total=len(s_vecs)):
            accs = fl_run_with_fixed_share(fit_helper, s_vec)
            all_accs[s_idx] = [accs[i] for i in range(types_of_data)]
        for i in range(types_of_data):
            try:
                popt, _ = curve_fit(
                    accuracy_func,
                    [
                        [s_vecs[j][i] for j in range(len(s_vecs))]
                        for i in range(types_of_data)
                    ],
                    all_accs[:, i],
                    np.zeros(types_of_data),
                )
                W[i] = popt
            except:
                print("Fail to fit curve")
        fl_results["W"] = W.tolist()
        with open("out/{}_non_iid_fl_weights.json".format(params["task"]), "w") as f:
            json.dump(fl_results, f, indent=4)

    logger.info("Fitted weights: {}".format(W))

    # set the rotation angles for each participant
    if rotation_angles is not None:
        type_of_agent = lambda x: int(
            (x / params["fl_total_participants"]) * types_of_data
        )
        params["rotation_angles"] = [
            rotation_angles[type_of_agent(i)]
            for i in range(params["fl_total_participants"])
        ]
    main_hlpr = Helper(params)
    costs = [
        random.uniform(0, 0.001) for _ in range(len(main_hlpr.task.all_users()))
    ]  # random costs for each user
    print(costs)
    print([len(user.train_loader) for user in main_hlpr.task.all_users()])
    fl_results["costs"] = costs
    fl_results["W"] = W.tolist()
    for m in ["br", "br-bg", "br-shap"]:
        main_hlpr.params.method = m
        logger.info("Running method: {}".format(m))
        logger.warning("Begin best response calculation for {}!".format(m))
        # s_vec = []
        s_vec = best_response(main_hlpr, W, costs)
        logger.warning("Finish best response calculation for {}!".format(m))
        logger.warning(
            "BE: [{}]".format(", ".join(["{:.2f}".format(float(v)) for v in s_vec]))
        )

        # accs = []
        accs = fl_run_with_fixed_share(main_hlpr, s_vec, verbose=True)
        fl_results[m] = {
            "BE": s_vec,
            "Acc": accs,
            "Costs": [costs[i] * s_vec[i] for i in range(len(s_vec))],
            "Sum of s": sum(s_vec),
        }
    with open("out/{}_non_iid_fl_results.json".format(params["task"]), "w") as f:
        json.dump(fl_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="params", default="fedavg.yaml")
    parser.add_argument("--name", dest="name", default="test", help="Tensorboard name")
    parser.add_argument(
        "--method", dest="method", default="br-shap", help="Tensorboard name"
    )
    parser.add_argument(
        "--idtest",
        dest="idtest",
        action="store_true",
        default=False,
        help="whether to use identical testing data",
    )

    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params["current_time"] = datetime.now().strftime("%b.%d_%H.%M.%S")
    params["name"] = args.name
    params["method"] = args.method
    params["idtest"] = args.idtest
    params["random_seed"] = random.randint(0, 10)

    non_iid_main(params, rotation_angles=[10, 90, 180])
    ## Make all the testing dataset the same
    # if args.idtest:
    #     logger.info("All the testing data are made identical!")
    #     helper.task.merge_test_data()

    # try:
    #     fl_run(helper)
    # except (KeyboardInterrupt):
    #     logger.error(f"Fine. Deleted: {helper.params.folder_path}")
    #     shutil.rmtree(helper.params.folder_path, ignore_errors=True)
    #     if helper.params.tb:
    #         shutil.rmtree(f'runs/{args.name}')
