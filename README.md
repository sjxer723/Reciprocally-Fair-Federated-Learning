# You Get What You Give: Reciprocal Fair Federated Learning

This repo contains the source codes of our work "You Get What You Give: Reciprocal Fair Federated Learning". The codes for running the dataset MNIST and CIFAR-10 are respectively put into the directories `MNIST/` and `CIFAR-10/`. Our codes is based on the open-sourced framework [backdoor101](https://github.com/ebagdasa/backdoors101).

## Installation
First instance the dependencies by `pip install -r requirements.txt`.
For each of the two directories, create two directories: `runs/` and `saved_models/`

## Repeating Experiments
Our expermental results are put into the directory `output/`. You can also repeat the experments by the following commands:

For MNIST,

* Run the FedBR method,
    ```bash
    cd MNIST/
    python3 main.py --method br
    ```
* Run the FedBR-BG method,
    ```bash
    cd MNIST/
    python3 main.py --method br-bg
    ```
* Run the FedBR-Shap method,
    ```bash
    cd MNIST/
    python3 main.py --method br-shap
    ```

For CIFAR-10,
* Run the FedBR method,
    ```bash
    cd CIFAR-10/
    python3 training.py --name cifar --params configs/cifar_fed.yaml --m br
    ```
* Run the FedBR-BG method,
    ```bash
    cd CIFAR-10/
    python3 training.py --name cifar --params configs/cifar_fed.yaml --m br-bg
    ```
* Run the FedBR-Shap method,
    ```bash
    cd CIFAR-10/
    python3 training.py --name cifar --params configs/cifar_fed.yaml --m br-shap
    ```

