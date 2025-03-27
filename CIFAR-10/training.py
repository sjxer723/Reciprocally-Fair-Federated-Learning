import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm
import random

# noinspection PyUnresolvedReferences
from shapley_value import FLInstance
from helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')
alpha_1 = 1.407
beta_1 = 0.281
alpha_2 = 1.256
beta_2 = 0.246
cost = [random.uniform(0.0004, 0.0005) for _ in range(100)]
cost_scalar_beta = 0.8
delta = 1

####### Accuracy Functions
def a(s_all: list):
    return 1 - (alpha * 1.0) / pow(sum(s_all), beta)

def a_derivative(s_all: list, alpha, beta):
    return (alpha * beta * 1.0) / pow(sum(s_all), beta+1)

def train_with_num_batch(hlpr: Helper, epoch, model, optimizer, train_loader, num_batch, attack=True):
    criterion = hlpr.task.criterion
    model.train()

    index_list = list(range(0, len(train_loader)))
    random_batch = random.sample(index_list, num_batch)

    j = 0
    with tqdm(total=num_batch, desc="Batch finshed: ") as pbar:
        for i, data in enumerate(train_loader):
            if i not in random_batch:
                continue
            batch = hlpr.task.get_batch(i, data)
            model.zero_grad()
            loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
            loss.backward()
            optimizer.step()
            # hlpr.report_training_losses_scales(j, epoch)
            pbar.update(1) 
            if i == hlpr.params.max_batch_id:
                break
            j += 1
    return

def train(hlpr: Helper, epoch, model, optimizer, train_loader, s_i, attack=True):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in tqdm(enumerate(train_loader)):
        if i >= s_i:
            break
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        print("Length of Loader", len(hlpr.task.test_loader))
        for i, (data, _) in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric

def test_num_batch(hlpr: Helper, epoch, num_batch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        print("Length of Loader", len(hlpr.task.test_loader))
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            if i >= num_batch:
                break
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric

def test_local(hlpr:Helper, model, test_loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            batch = hlpr.task.get_batch(i, data)
            outputs = model(batch.inputs)
            labels = batch.labels
            _, predicted = torch.max(outputs, dim=1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions * 1.0 /total_samples
    print("{} correct of {}, Accuracy: {}".format(correct_predictions, total_samples, correct_predictions * 1.0 /total_samples))
    return accuracy 

def run(hlpr, num_batch):
    train_loader = hlpr.task.train_loader
    logger.warning("Begin training with {} batches data".format(num_batch))
    for epoch in range(hlpr.params.start_epoch,
                    hlpr.params.epochs + 1):
        train_with_num_batch(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
            train_loader, num_batch)
    acc = test_num_batch(hlpr, epoch, 50, backdoor=False)
    logger.warning("Batch: {}, Acc: {}".format(num_batch, acc))

def fl_run(hlpr: Helper):
    s_dict = {}
    for user_id in range(hlpr.params.fl_total_participants):
        s_dict[user_id] = 1.0   # initialized s_i as 1 for each i

    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        accs, s_dict = run_fl_round(hlpr, epoch, s_dict)
        metric = test(hlpr, epoch, backdoor=False)
        # test(hlpr, epoch, backdoor=True)

        # hlpr.save_model(hlpr.task.model, epoch, metric)
        logger.warning("Epoch {}".format(epoch))
        logger.warning("The sum of accs: {}".format(sum(accs.values())))
        logger.warning("s_i: {}".format(s_dict))

def run_fl_round(hlpr, epoch, s_dict):
    global alpha, beta

    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()
    accs = {}
    fl = FLInstance(len(s_dict), list(s_dict.values()), alpha_1, beta_1, 0.01)

    for user in tqdm(round_participants):
        s = s_dict[user.user_id]
        if user.user_id <= len(round_participants)/2:
            fl.alpha = alpha_1
            fl.beta = beta_1
        else:
            fl.alpha = alpha_2
            fl.beta = beta_2
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        accuracy = test_local(hlpr, local_model, user.test_loader)
        accs[user.user_id] = accuracy
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, s, attack=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, s, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

        ## Update s_i
        if hlpr.params.method == "br-shap":
            # FedBR-SV, ∂d/∂s = ∂φ/∂s - c
            # print(fl.compute_shapley_value_derivative(self.id), self.cost_gradient(s))
            d = fl.compute_shapley_value_derivative(user.user_id) - cost[user.user_id]
        elif hlpr.params.method == "br":
            # FedBR, ∂d/∂s = ∂a/∂s - c
            d = a_derivative(s_dict.values(), fl.alpha, fl.beta) - cost[user.user_id]
        elif hlpr.params.method == "br-bg":
            # FedBR-BG, ∂d/∂s = ∂a/∂s - (1-β)*c
            d = a_derivative(s_dict.values(), fl.alpha, fl.beta) - (1 - cost_scalar_beta) * cost[user.user_id]
        else:
            d = 0
        s_ =  s + delta * d
        if not (s_ > len(user.train_loader) or s_ < 0):
            s = s_
        s_dict[user.user_id] = s

    hlpr.task.update_global_model(weight_accumulator, global_model)
    
    return accs, s_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--method', dest='method', default='br-shap', help='Tensorboard name')
    parser.add_argument('--batch', dest='batch', default=10)

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name
    params['method'] = args.method

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            run(helper, int(args.batch))
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
