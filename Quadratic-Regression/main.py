import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np

y_threshold = 0.5

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

    def train(self, global_model, epochs=10, lr=0.005):
        local_model = copy.deepcopy(global_model)
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = local_model(self.train_X)
            loss = loss_fn(y_pred, self.train_y)
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

def fl_run(num_rounds=300, local_epochs=5, lr=0.001):
    A = np.random.randn(10, 10)
    A = (A + A.T) / 2 
    B = np.random.randn(10)
    c = np.random.randn()

    client1 = Client(A, B, c, training_size=1000, testing_size=100, label=1)
    client2 = Client(A, B, c, training_size=1000, testing_size=100, label=0)

    global_model = QuadraticClassifier()
    print("Initial model parameters:")

    for round in range(num_rounds):
        params_a, loss_a = client1.train(global_model, epochs=local_epochs, lr=lr) 
        params_b, loss_b = client2.train(global_model, epochs=local_epochs, lr=lr)         

        with torch.no_grad():
            for param_g, param_a, param_b in zip(global_model.parameters(), params_a.values(), params_b.values()):
                param_g.copy_((param_a + param_b) / 2)

        if round % 10 == 0:
            acc_a = client1.compute_accuracy(global_model)
            acc_b = client2.compute_accuracy(global_model)
            print(f"Round {round}: Accuracy A = {acc_a:.2%}, Accuracy B = {acc_b:.2%}")

    return global_model

if __name__ == "__main__":
    global_model = fl_run()
    print("Parameters of the global model:")
    for name, param in global_model.named_parameters():
        print(f"{name}: {param.data.numpy()}")