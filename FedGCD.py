import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from math import log
import numpy as np
from leaf.dataloader import FEMNISTDataset
from torch.utils.data import DataLoader

class Client:
    def __init__(self, dataset, model, device, lr=0.01, batch_size=32):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

    def learn(self):
        self.model.train()
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()

    def send_model(self):
        return self.model.state_dict()

    def receive_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

class Server:
    def __init__(self, num_clients, model, device, initial_community_weights=None):
        self.num_clients = num_clients
        self.model = model
        self.device = device
        self.client_models = [model.state_dict() for _ in range(self.num_clients)]
        if initial_community_weights is not None:
            self.community_weights = initial_community_weights
        else:
            self.community_weights = np.ones(num_clients)

    def create_adjacency_matrix(self, clients):
        client_weights = [client.send_model() for client in clients]
        adjacency_matrix = np.zeros((self.num_clients, self.num_clients))

        for i in range(self.num_clients):
            for j in range(self.num_clients):
                adjacency_matrix[i, j] = torch.tensor([torch.norm(client_weights[i][key] - client_weights[j][key]).item() for key in client_weights[i].keys()]).sum().item()
        return adjacency_matrix

    def find_optimal_num_communities(self, adjacency_matrix, max_communities=):
        best_aic = float("inf")
        best_num_communities = 1
        num_samples = adjacency_matrix.shape[0]

        for n in range(1, max_communities + 1):
            nmf_model = NMF(n_components=n, init='random', random_state=0)
            W = nmf_model.fit_transform(adjacency_matrix)
            H = nmf_model.components_
            reconstructed_matrix = np.dot(W, H)
            mse = mean_squared_error(adjacency_matrix, reconstructed_matrix)
            k = n * (num_samples + 1)
            aic = num_samples * log(mse) + 2 * k

            if aic < best_aic:
                best_aic = aic
                best_num_communities = n

        return best_num_communities

    def detect_communities(self, clients, max_communities):
        adjacency_matrix = self.create_adjacency_matrix(clients)
        num_communities = self.find_optimal_num_communities(adjacency_matrix, max_communities)
        model = NMF(n_components=num_communities, init='random', random_state=0)
        W = model.fit_transform(adjacency_matrix)
        H = model.components_
        kmeans = KMeans(n_clusters=num_communities, random_state=0).fit(W)
        return kmeans.labels_, W

    def federated_averaging(self, clients, community_labels, W):
        num_communities = np.unique(community_labels).shape[0]
        for community in range(num_communities):
            community_indices = [i for i, label in enumerate(community_labels) if label == community]
            community_client_models = [clients[i].send_model() for i in community_indices]
            community_weights = self.calculate_weights(W, community_indices)
            weighted_avg = self.weighted_average(community_client_models, community_weights)
            for i in community_indices:
                clients[i].receive_model(weighted_avg)

    def calculate_weights(self, W, community_indices):
        community_weights = W[community_indices]
        normalized_scores = community_weights / np.sum(community_weights, axis=1, keepdims=True)
        client_weights = np.dot(normalized_scores, self.community_weights)
        return client_weights

    def weighted_average(self, client_models, weights):
        avg = client_models[0].copy()
        for key in avg.keys():
            avg[key] *= 0

        for i, model in enumerate(client_models):
            for key in model.keys():
                avg[key] += weights[i] * model[key]

        return avg

    def evaluate(self, clients):
        self.model.eval()
        correct = 0
        total = 0
        for client in clients:
            for data, target in client.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total


def load_femnist_data(num_clients):
    femnist_data = FEMNISTDataset("data", num_clients)
    return [DataLoader(femnist_data.clients[i], batch_size=32, shuffle=True) for i in range(num_clients)]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 100
    max_communities = 2
    num_rounds = 200
    clustering_interval = 50
    performance_interval = 10

    # Load FEMNIST data
    client_dataloaders = load_femnist_data(num_clients)

    # Define the model architecture
    # Replace GraphNeuralNetwork with the appropriate model for FEMNIST, e.g., CNN or MLP
    model = CNNModel().to(device)

    # Initialize clients
    clients = [Client(client_dataloaders[i], model, device) for i in range(num_clients)]

    # Initialize the server
    server = Server(num_clients, model, device, num_communities, initial_community_weights=first_learning_weights)

    # Federated learning process
    for round in range(1, num_rounds + 1):
        print(f"Round {round}")

        # Clients learn locally
        for client in clients:
            client.learn()

        # Cluster clients and apply FedAvg every clustering_interval rounds
        if round % clustering_interval == 1:
            community_labels, W = server.detect_communities(clients,max_communities)
            server.federated_averaging(clients, community_labels, W)

        # Output performance every performance_interval rounds
        if round % performance_interval == 0:
            accuracy = server.evaluate(clients)
            print(f"Performance at round {round}: {accuracy:.4f}")

if __name__ == "__main__":
    main()
