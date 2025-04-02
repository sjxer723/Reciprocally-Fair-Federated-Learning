import torch
import copy
import itertools
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from scipy.optimize import curve_fit

## Basic configurations
n = 2                   # number of clients
num_of_labels = 2       # number of labels of the classification task
y_threshold = 0.5       # threshold for the label
s_eps = 50              # each client add s_eps more samples to estimate the gradient of shapley share
s_lr = 1000             # learning rate for the share update
costs = np.random.uniform(0, 0.001, n)
training_size=1000      # number of training samples
fit_sample_delta=100    # delta value for fitting the accuracy function
testing_size=100        # number of testing samples
cost_scalar_beta=0.8    # beta value for br-bg method

def generate_random_quadratic_data(A, B, c, num_points=100, delta=0.5, label=1):
    # X = np.random.uniform(-10, 10, (num_points, 10))
    # Y_curve = np.sum(X @ A * X, axis=1) + np.dot(X, B) + c      
    x_points = []
    while len(x_points) < num_points:
        x = np.random.uniform(-10, 10, (1, 10))
        y = (np.sum(x @ A * x, axis=1) + np.dot(x, B) + c)[0]
        x = x[0]
        ## if label is 1, then most y should be positive and vice versa
        if label == 1:
            if abs(y) > y_threshold and len(x_points) < num_points * 0.8:
                x_points.append(x)
            elif len(x_points) >= num_points * 0.8:
                x_points.append(x)
            else:
                continue
        else:
            if abs(y) < y_threshold and len(x_points) < num_points * 0.8:
                x_points.append(x)
            elif len(x_points) >= num_points * 0.8:
                x_points.append(x)
            else:
                continue
                        
    X = np.array(x_points)
    Y = np.sum(X @ A * X, axis=1) + np.dot(X, B) + c  
    # Add some noise to the y values, since the real distribution may not be perfect quadratic
    noise = np.random.normal(-delta, delta, num_points)
    Y = Y + noise
    labels = (Y >= y_threshold).astype(float).reshape(-1, 1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

class QuadraticClassifier(nn.Module):
    def __init__(self):
        super(QuadraticClassifier, self).__init__()
        self.A = nn.Parameter(torch.randn(10, 10) * 0.01) 
        self.B = nn.Parameter(torch.randn(10, 1) * 0.01)
        self.poly = nn.Linear(13, 1)
        
    def forward(self, x):
        quad_features = torch.cat([
            (x @ self.A * x).sum(dim=1, keepdim=True),
            x @ self.B,
            x,
            torch.ones_like(x[:, :1])
        ], dim=1)

        return torch.sigmoid(self.poly(quad_features))

class Client():
    def __init__(self, A, B, c, training_size=100, testing_size=50, noise_std=1.0, label=1):
        self.train_X, self.train_y = generate_random_quadratic_data(A, B, c, num_points=training_size, delta=1.0, label=label)
        self.test_X, self.test_y = generate_random_quadratic_data(A, B, c, num_points=testing_size, delta=1.0, label=label)

    def train(self, global_model, epochs=10, lr=0.005, share_size=10):
        local_model = copy.deepcopy(global_model)
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = local_model(self.train_X[:share_size, :])
            loss = loss_fn(y_pred, self.train_y[:share_size])
            loss.backward()
            optimizer.step()
        
        return local_model.state_dict(), loss.item()

    def compute_accuracy(self, model):
        with torch.no_grad():
            y_pred = model(self.test_X)
            y_pred_labels = (y_pred >= y_threshold).float()
            correct = (y_pred_labels == self.test_y).sum().item()
            total = self.test_y.size(0)
            return correct / total

class GroundTruth():
    def __init__(self):
        self.A = np.random.randn(10, 10)
        self.A = (self.A + self.A.T) / 2 
        self.B = np.random.randn(10)
        self.c = np.random.randn()
    
    def parameters(self):
        return self.A, self.B, self.c

## Protocol 0: FL with fixed shares
def fl_run(clients, s_vec, num_rounds=300, local_epochs=5, lr=0.001, verbose=False):
    global_model = QuadraticClassifier()
    for round in range(num_rounds):
        params_all = []
        for i, client in enumerate(clients):
            params, _ = client.train(global_model, epochs=local_epochs, lr=lr, share_size=s_vec[i])
            params_all.append(params.values())
        
        with torch.no_grad():
            for param_g, *params in zip(global_model.parameters(), *params_all):
                param_g.copy_(sum(params) / n)

        if round % 10 == 0:
            accs = []
            for client in clients:
                acc = client.compute_accuracy(global_model)
                accs.append(acc)
            if verbose:
                print(f"Round {round}: Accuracy {accs}, Shares {s_vec}")

    return global_model, accs

## Protocol 1: FL without closed form
def fl_run_without_closed_form(clients: list[Client], num_rounds=300, local_epochs=5, lr=0.001, verbose=False):
    # initial data shares 
    s_vec = [10 for _ in range(n)]
    assert(all([s > 0 for s in s_vec]))
    
    global_model = QuadraticClassifier()
    
    for round in range(num_rounds):
        params_all = []
        params_eps_all = []

        for i, client in enumerate(clients):
            params, _ = client.train(global_model, epochs=local_epochs, lr=lr, share_size=s_vec[i])
            params_all.append(params.values())
            params_eps, _ = client.train(global_model, epochs=local_epochs, lr=lr, share_size=s_vec[i] + s_eps)
            params_eps_all.append(params_eps.values())
        
        all_permutations = list(itertools.permutations(range(n)))
        gradients = np.zeros(n)
        for perm in all_permutations:
            for i in range(n):
                # get model T_i
                model_fst_i = copy.deepcopy(global_model)
                with torch.no_grad():
                    for param_g, *param_i in zip(model_fst_i.parameters(), *params_all):
                        # print(sum(param_i[:i+1]))
                        param_g.copy_(sum([param_i[perm[j]] for j in range(i)]) / (i + 1))
                # get model T_i^eps
                model_eps_fst_i = copy.deepcopy(global_model)
                with torch.no_grad():
                    for param_g, *param_i in zip(model_eps_fst_i.parameters(), *params_all, *params_eps_all):
                        param_g.copy_(((sum([param_i[perm[j]] for j in range(i-1)])) + param_i[n + perm[i]]) / (i + 1))
                
                ## estimate the gradient
                accs, accs_eps = [], []
                for client in clients:
                    acc = client.compute_accuracy(model_fst_i)
                    acc_eps = client.compute_accuracy(model_eps_fst_i)
                    accs.append(acc)
                    accs_eps.append(acc_eps)
                gradients[perm[i]] += (sum(accs_eps) - sum(accs)) / s_eps
                
        gradients /= len(all_permutations)
        for i, s, g in zip(list(range(n)), s_vec, gradients):
            s_updated = int(s + s_lr * (g - costs[i]))
            if s_updated >= training_size or s_updated <= 0:
                continue
            else:
                s_vec[i] = s_updated

        if verbose:
            print(gradients, s_vec)
        
        ## Update the global model
        with torch.no_grad():
            for param_g, *params in zip(global_model.parameters(), *params_all):
                param_g.copy_(sum(params) / n)
        
        if round % 10 == 0:
            accs = []
            for client in clients:
                acc = client.compute_accuracy(global_model)
                accs.append(acc)
            if verbose:
                print(f"Round {round}: Accuracy {accs}, Shares {s_vec}")

    return global_model, accs, s_vec

# Protcol 2: FL with closed form
def accuracy_func(S, *w):
    len_of_S = len(S)
    return 1 - 1 / (1 + sum([w[i] * S[i] for i in range(len_of_S)]))

def derivate_of_accuracy(S, *w, j):
    len_of_S = len(S)
    return w[j] / (1 + sum([w[i] * S[i] for i in range(len_of_S)]))**2

def fit_closed_form_accuracy(clients: list[Client], num_rounds=300, local_epochs=5, lr=0.001, verbose=False):
    print("Begin fitting the closed form accuracy function!")
    num_of_samples_per_dimension = training_size // fit_sample_delta
    s_vecs = list(itertools.product(range(num_of_samples_per_dimension + 1), repeat=n))
    all_accs = np.zeros((len(s_vecs), n))

    for s_idx, s_vec in enumerate(s_vecs):
        _, accs = fl_run(clients, s_vec, num_rounds=num_rounds // 10, local_epochs=local_epochs, lr=lr, verbose=False)
        all_accs[s_idx] = accs
    if verbose:
        print(all_accs)

    w = np.zeros((n, n))
    for i in range(n):
        try:
            popt, _ = curve_fit(accuracy_func, 
                                [[s_vecs[j][i] for j in range(len(s_vecs))] for i in range(n)],
                                all_accs[:, i], np.zeros(n))
            w[i] = popt
            if verbose:
                print("popt:", popt)
        except:
            print("Fail to fit curve")
    print("Finish fitting the closed-form accuracy function!", w)

    return w

def fl_run_with_closed_form(clients: list[Client], w, method:str, num_rounds=300, local_epochs=5, lr=0.001, verbose=False):
    global_model = QuadraticClassifier()
    s_vec = [10 for _ in range(n)]

    def update_share(S, w, i):
        derivate = derivate_of_accuracy(S, *w, j=i)
        if method == "br":
            s_updated = int(S[i] + s_lr * (derivate - costs[i]))
        elif method == "br-bg": 
            s_updated = int(S[i] + s_lr * (derivate - (1 - cost_scalar_beta) * costs[i]))
        if s_updated >= training_size or s_updated <= 0:
            return S[i]
        else:
            return s_updated

    for round in range(num_rounds):
        params_all = []
        for i, client in enumerate(clients):
            params, _ = client.train(global_model, epochs=local_epochs, lr=lr, share_size=s_vec[i])
            params_all.append(params.values())
            s_vec[i] = update_share(s_vec, w[i], i)

        with torch.no_grad():
            for param_g, *params in zip(global_model.parameters(), *params_all):
                param_g.copy_(sum(params) / n)

        if round % 10 == 0:
            accs = []
            for client in clients:
                acc = client.compute_accuracy(global_model)
                accs.append(acc)
            if verbose:
                print(f"Round {round}: Accuracy {accs}, Shares {s_vec}")

    return global_model, accs, s_vec

def _main_for_synthetic_data():
    g = GroundTruth()
    A, B, c = g.parameters()
    results_json = {}
    clients = []
    for i in range(num_of_labels):
        client = Client(A, B, c, training_size=training_size, testing_size=testing_size, label=i)
        clients.append(client)

    ## Run the FL protocol without closed form
    global_model, accs, s_vec = fl_run_without_closed_form(clients)
    cost_of_clients = [costs[i] * s_vec[i] for i in range(n)]
    print("Accuracy of Fed-Shap: ", accs)
    print("Shares of Fed-Shap  :", s_vec)
    print("Costs  of Fed-Shap  :", cost_of_clients)
    print("Welfare of Fed-Shap :", sum(accs) - sum(cost_of_clients))
    results_json['br-shap'] = {
        "Acc": accs,
        "BE": s_vec,
        "Costs": cost_of_clients,
        "Sum of s": sum(s_vec),
    }
    ## Run the FL protocol with closed form
    w = fit_closed_form_accuracy(clients)     # find the fitted closed form accuracys
    results_json['W'] = w.tolist()
    for method in ["br", "br-bg"]:
        global_model, accs, s_vec = fl_run_with_closed_form(clients, w, method)
        cost_of_clients = [costs[i] * s_vec[i] for i in range(n)]
        print("Accuracy of  {}: {}".format(method, accs))
        print("Shares of    {}: {}".format(method, s_vec))
        print("Costs  of    {}: {}".format(method, cost_of_clients))
        print("Welfare of   {}: {}".format(method, sum(accs) - sum(cost_of_clients)))
        results_json[method] = {
            "Acc": accs,
            "BE": s_vec,
            "Costs": cost_of_clients,
            "Sum of s": sum(s_vec),
        }
    with open("quad_fed.json", 'w') as f:
        json.dump(results_json, f)

if __name__ == "__main__":
    _main_for_synthetic_data()
