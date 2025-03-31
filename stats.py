import os
import json
import pandas as pd
from MNIST.shapley_value import FLInstance

def statistics(path, prefix, name):
    all_results = {m: [] for m in ["br", "br-bg", "br-shap"]}
    avg_statistics, median_statistics = [], []
    num_of_attempts = 3
    num_of_clients = 30
    num_of_types = 3
    cost_scalar_beta = 0.8

    for i in range(1, 1+num_of_attempts):
        result_file_path = os.path.join(path, f"{prefix}{i}.json")
        try:
            with open(result_file_path, 'r') as f:
                data = json.load(f)
                costs = data.get("costs", [])
                W = data.get("W", [])
        except:
            print(f"Error reading {result_file_path}. Please run experiment first!")
        def accuracy_func_of_i(S, i):
            group_of_i = int(i / num_of_clients * num_of_types)  # Determine the group of the agent
            denominator = 1
            for j in range(num_of_clients):
                group_of_j = int(j / num_of_clients * num_of_types)  # Determine the group of the agent
                denominator += W[group_of_i][group_of_j] * S[j]
 
            return 1 - 1.0 / denominator
 
        def accuracy_func(S, clients):
            accs = 0
            for t in range(num_of_types):
                denominator = 1
                for i in clients:
                    group = int(i / num_of_clients * num_of_types)  # Determine the group of the agent
                    denominator += W[t][group] * S[i]
                accs += (1 - 1.0 / denominator) * (num_of_clients // num_of_types) 
            return accs
        
        for m in ["br", "br-bg", "br-shap"]:
            result_of_m = data.get(m, None)
            be = result_of_m.get("BE", [])
            FL = FLInstance(num_of_clients, be, 0, 0, _eps=0.1)  # alpha and beta are not used here
            shapley_shares = FL.compute_shapley_values(lambda clients: accuracy_func(be, clients))
            if m == "br":
                reciprocities = [accuracy_func_of_i(be, i) / shapley_shares[i] for i in range(num_of_clients)]
            elif m == "br-bg":
                payments = [cost_scalar_beta * (costs[i] - sum([costs[j] for j in range(num_of_clients) if j!=i]) / (num_of_clients - 1))
                                             for i in range(num_of_clients)]
                reciprocities = [(accuracy_func_of_i(be, i) + payments[i])/ shapley_shares[i] for i in range(num_of_clients)]
            elif m == "br-shap":
                reciprocities = [1 for _ in range(num_of_clients)]

            reciprocity = min(reciprocities)          
            all_results[m].append({
                "Data Shares": result_of_m.get("Sum of s"),
                "Accuracy": sum(result_of_m.get("Acc", 0.0)) / num_of_clients,  
                "Welfare":  (sum(result_of_m.get("Acc", [])) - sum(result_of_m.get("Costs", []))) \
                            / num_of_clients,
                "Reciprocity": reciprocity,
            })
    for i in range(num_of_attempts):
        for m in ['br', 'br-bg', 'br-shap']:
            if m == 'br':
                all_results[m][i]['DataGain'] = 1
                all_results[m][i]['AccGain'] = 1
            else:
                all_results[m][i]['DataGain'] = all_results[m][i]['Data Shares'] * 1.0 / all_results['br'][i]['Data Shares']
                all_results[m][i]['AccGain'] = all_results[m][i]['Accuracy'] * 1.0 / all_results['br'][i]['Accuracy']

    for m in ["br", "br-bg", "br-shap"]:
        # print(all_results[m][0])
        result_for_m = dict()
        result_for_m["Benchmark"] = name
        result_for_m["Method"] = m
        for key in all_results[m][0].keys():
            result_for_m[key] = sum([all_results[m][i][key] for i in range(num_of_attempts)]) / num_of_attempts
        avg_statistics.append(result_for_m)

        result_for_m = dict()
        result_for_m["Benchmark"] = name
        result_for_m["Method"] = m
        for key in all_results[m][0].keys():
            result_for_m[key] = sorted([all_results[m][i][key] for i in range(num_of_attempts)])[1]
        median_statistics.append(result_for_m)
    
    avg_df = pd.DataFrame(avg_statistics)
    avg_df = avg_df.round(3)
    median_df = pd.DataFrame(median_statistics)
    median_df = median_df.round(3)

    return avg_df, median_df

info = [
    ("CIFAR-10/out", "CifarFed_non_iid_fl_results", "CIFAR-10"),
    ("MNIST/out/FashionMNIST", "FashionMNIST_FedAvg_non_iid_fl_results", "FashionMNIST"),
    ("MNIST/out/MNIST", "MNIST_FedAvg_non_iid_fl_results", "MNIST")
]

for path, prefix, name in info:
    avg_df, median_df = statistics(path, prefix, name)
    avg_df.to_csv(os.path.join(path, f"{prefix}_avg_statistics.csv"), index=False)
    median_df.to_csv(os.path.join(path, f"{prefix}_median_statistics.csv"), index=False)

    print(f"Average Statistics for {name}:")
    print(avg_df)
    print(f"Median Statistics for {name}:")
    print(median_df)
    print("--------------------------------------------------")