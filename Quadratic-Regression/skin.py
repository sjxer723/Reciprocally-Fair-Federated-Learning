import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, BatchNormalization
import os, math
import argparse
import random
import itertools
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit


def GELU(x):
    return 0.5 * x * (1 + tf.nn.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x**3))))


class ResMLPBlock(tf.keras.layers.Layer):
    def __init__(self, units, residual_path):
        super(ResMLPBlock, self).__init__()
        self.residual_path = residual_path
        self.D1 = Dense(units, activation="relu")
        self.D2 = Dense(units, activation="relu")
        if self.residual_path:
            self.D3 = Dense(units)
            self.D4 = Dense(units)

    def call(self, inputs):
        residual = inputs
        x = self.D1(inputs)
        y = self.D2(x)
        if self.residual_path:
            residual = self.D3(inputs)
            residual = GELU(residual)
            residual = self.D4(residual)
            residual = GELU(residual)
        return y + residual


class ResMLP(tf.keras.Model):
    def __init__(self, initial_filters, block_list, num_classes):
        super(ResMLP, self).__init__()
        self.initial_filters = initial_filters
        self.D1 = Dense(self.initial_filters, activation="relu")
        self.B1 = BatchNormalization()
        self.blocks = tf.keras.models.Sequential()
        for block_id, layers in enumerate(block_list):
            for layer_id in range(layers):
                residual = block_id != 0 and layer_id == 0
                self.blocks.add(ResMLPBlock(self.initial_filters, residual))
            self.initial_filters *= 2
        self.D2 = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.D1(inputs)
        x = self.B1(x)
        x = self.blocks(x)
        return self.D2(x)


def test(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    correct = np.sum(y_pred == y_test.numpy())
    total = len(y_test)
    accuracy = correct / total
    return accuracy


df = pd.read_csv("./Lumpy skin disease data.csv")
df = df.drop(
    ["region", "country", "reportingDate", "X5_Ct_2010_Da", "X5_Bf_2010_Da"], axis=1
)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values


def create_data(
    X, y, sample_size=10000, test_size=0.2, random_state=42, weights=[0.5, 0.5]
):
    np.random.seed(random_state)
    indices_class_0, indices_class_1 = np.where(y == 0)[0], np.where(y == 1)[0]

    indices1 = np.random.choice(
        indices_class_0, size=int(sample_size * weights[0]), replace=True
    )
    indices2 = np.random.choice(
        indices_class_1, size=int(sample_size * weights[1]), replace=True
    )
    indices = np.concatenate((indices1, indices2))

    X_sample, y_sample = X[indices], y[indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=test_size,
        random_state=random_state,
        stratify=y_sample,
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        tf.cast(X_train, tf.float64),
        tf.cast(X_test, tf.float64),
        tf.cast(y_train, tf.int64),
        tf.cast(y_test, tf.int64),
    )


max_size = 2000
n = 2  # number of players
num_of_groups = 2
X_train_A, X_test_A, y_train_A, y_test_A = create_data(
    X, y, sample_size=max_size, weights=[0.3, 0.7]
)
X_train_B, X_test_B, y_train_B, y_test_B = create_data(
    X, y, sample_size=max_size, weights=[0.7, 0.3]
)

player_A = ResMLP(initial_filters=32, block_list=[2, 2, 2, 2], num_classes=2)
player_B = ResMLP(initial_filters=32, block_list=[2, 2, 2, 2], num_classes=2)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
player_A.compile(optimizer="adam", loss=loss_fn)
player_B.compile(optimizer="adam", loss=loss_fn)

epochs_per_round = 1
num_rounds = 10
s_lr = 1000
T = 2000  # iterations of best response
cost_scalar_beta = 0.8


def run_fl_with_fixed_share(s_vec, _num_rounds=10, verbose=False):
    for round_num in range(_num_rounds):
        if verbose:
            print(f"Federated Learning Round {round_num + 1}/{num_rounds}")

        s1, s2 = int(s_vec[0]), int(s_vec[1])
        if s1:
            player_A.fit(
                X_train_A[:s1],
                y_train_A[:s1],
                batch_size=16,
                epochs=epochs_per_round,
                verbose=0,
            )
        if s2:
            player_B.fit(
                X_train_B[:s2],
                y_train_B[:s2],
                batch_size=16,
                epochs=epochs_per_round,
                verbose=0,
            )

        new_weights = [
            (a + b) / 2 for a, b in zip(player_A.get_weights(), player_B.get_weights())
        ]

        player_A.set_weights(new_weights)
        player_B.set_weights(new_weights)

        acc_A = test(player_A, X_test_A, y_test_A)
        acc_B = test(player_B, X_test_B, y_test_B)
        if verbose:
            print(
                f"Round {round_num + 1} - Player A Accuracy: {acc_A}, Player B Accuracy: {acc_B}"
            )

    return [acc_A, acc_B]


def accuracy_func(S, *w):
    len_of_S = len(S)
    return 1 - 1 / (1 + sum([w[i] * S[i] for i in range(len_of_S)]))


def derivate_of_accuracy(S, *w, j):
    len_of_S = len(S)
    return w[j] / (1 + sum([w[i] * S[i] for i in range(len_of_S)])) ** 2


def fit_closed_form_accuracy(fit_sample_delta=10240, n=2, verbose=False):
    print("Begin fitting the closed form accuracy function!")
    num_of_samples_per_dimension = max_size // fit_sample_delta
    s_vecs = list(
        itertools.product(
            [fit_sample_delta * i for i in range(num_of_samples_per_dimension + 1)],
            repeat=n,
        )
    )
    all_accs = np.zeros((len(s_vecs), n))

    for s_idx, s_vec in enumerate(s_vecs):
        print(f"Fitting for s_vec {s_vec} ({s_idx + 1}/{len(s_vecs)})")
        accs = run_fl_with_fixed_share(s_vec)
        all_accs[s_idx] = accs

    # print(s_vecs, all_accs)
    w = np.zeros((n, n))
    for i in range(n):
        try:
            popt, _ = curve_fit(
                accuracy_func,
                [[s_vecs[j][i] for j in range(len(s_vecs))] for i in range(n)],
                all_accs[:, i],
                np.zeros(n),
            )
            w[i] = popt
        except:
            print("Fail to fit curve")
    print("Finish fitting the closed-form accuracy function!", w)

    return w


def best_response(W, costs, method: str, verbose=False):
    s_vec = [1.0 for _ in range(n)]  # initial data share vector

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
            S, *W, j=i, grouping_fun=lambda x: int((x / n) * num_of_groups)
        )
        s_updated = S[i]
        if method == "br":
            s_updated = S[i] + s_lr * (derivative - costs[i])
        elif method == "br-bg":
            s_updated = S[i] + s_lr * (derivative - (1 - cost_scalar_beta) * costs[i])
        elif method == "br-shap":
            # Using Shapley value as the adjustment
            perms = list(itertools.permutations(range(n)))
            derivative_of_shapley = 0
            for perm in perms:
                id_of_i = perm.index(i)  # find the index of player i in the permutation
                agents_with_i = perm[: id_of_i + 1]
                derivative_of_shapley += partial_derivative_of_accuracy(
                    S,
                    *W,
                    agents_with_i=agents_with_i,
                    i=i,
                    grouping_fun=lambda x: int((x / n) * num_of_groups),
                )
            derivative_of_shapley /= len(perms)
            s_updated = S[i] + s_lr * (derivative_of_shapley - costs[i])
        else:
            raise ValueError("unknown method {} for best response".format(method))
        # print(s_updated)
        if s_updated >= max_size or s_updated <= 0:
            return S[i]
        else:
            return s_updated

    for epoch in range(T):
        ## Update share for every participant
        s_vec[0] = update_share(s_vec, W, 0)
        s_vec[1] = update_share(s_vec, W, 1)
        ## Report the accuracy for all testing datasets
        if epoch % 10 == 0 and verbose:
            incurred_costs = [c * s for c, s in zip(costs, s_vec)]
            print(
                "Epoch: {}, Sum of s_i: {}, Costs: {}".format(
                    epoch, sum(s_vec), sum(incurred_costs)
                )
            )

    return s_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skin disease forecasting")
    parser.add_argument("--n", type=int, default=1, help="number of run")

    args = parser.parse_args()
    results_json = {}
    W = fit_closed_form_accuracy(fit_sample_delta=500, n=2)
    costs = [random.uniform(0, 1e-7) for _ in range(n)]  # random costs for each user

    results_json["W"] = W.tolist()
    results_json["Costs"] = costs
    for method in ["br", "br-bg", "br-shap"]:
        s_vec = best_response(W, costs, method)
        print("Method: {}, BE: {}".format(method, s_vec))

        accs = run_fl_with_fixed_share(s_vec, _num_rounds=100)
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
    with open("skin_fed{}.json".format(args.n), "w") as f:
        json.dump(results_json, f)
