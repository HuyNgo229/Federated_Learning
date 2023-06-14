import warnings
from collections import OrderedDict
from typing import Dict, Tuple
import json
import os

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from utils.model import CNNModel

from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import argparse

FL_ROUND = 0
eval_list = []

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module(imported), train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, traindata, dataset, epochs, device) -> Tuple[float, float]:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    print(
        f"Training {dataset} dataset with {epochs} local epoch(s) w/ {len(traindata)} batches each"
    )

    # Train the network
    model.to(device)
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0.0
        correct = 0
        for i, data in enumerate(traindata, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = running_loss
            accuracy = correct / total
            if i == len(traindata) - 1:  # print every 100 mini-batches
                accuracy = correct / total
                loss_batch = running_loss / len(traindata)
                print(
                    f"Train Dataset {dataset} with [{epoch+1}, {i+1}] \
                    loss: {loss_batch} accuracy: {accuracy}"
                )
                running_loss = 0.0
        loss = loss / len(traindata)
    return loss, accuracy


def test(model, dataset, testdata, device) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    # Evaluate the network
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in testdata:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(testdata)
    print(f"Dataset {dataset} with evaluation loss: {loss}")
    return loss, accuracy

def load_datasets(dataset: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    BATCH_SIZE = 32
    NUM_CLIENTS = 100
    if dataset == "HBCDs":
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(), # randomly flip and rotate
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        data_dir = '/content/b_cancer_data'
        trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        testset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transforms['valid'])
    else:
        print("No valid dataset available")

    num_examples = {
        "dataset": dataset,
        "trainset": len(trainset),
        "testset": len(testset),
    }
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader, num_examples


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################




# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing image classification using PyTorch."""

    def __init__(
        self,
        model: CNNModel,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
        mode: str,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.mode = mode

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using
        BN layers."""
        self.model.train()
        # pylint: disable = no-else-return
        if self.mode == "fedbn":
            # Excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return all model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn
        layer if available."""
        self.model.train()
        # pylint: disable=not-callable
        if self.mode == "fedbn":
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        # pylint: enable=not-callable

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Set model parameters, train model, return updated model
        parameters."""
        self.set_parameters(parameters)
        test_loss, test_accuracy = test(
            self.model, self.num_examples["dataset"], self.trainloader, device=DEVICE
        )
        test_dict = {
            "dataset": self.num_examples["dataset"],
            "fl_round": FL_ROUND,
            "strategy": self.mode,
            "train_loss": test_loss,
            "train_accuracy": test_accuracy,
        }
        loss, accuracy = train(
            self.model,
            self.trainloader,
            self.num_examples["dataset"],
            epochs=1,
            device=DEVICE,
        )
        eval_list.append(test_dict)
        return (
            self.get_parameters({}),
            self.num_examples["trainset"],
            {"loss": loss, "accuracy": accuracy},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Set model parameters, evaluate model on local test dataset, return
        result."""
        self.set_parameters(parameters)
        global FL_ROUND
        loss, accuracy = test(
            self.model, self.num_examples["dataset"], self.testloader, device=DEVICE
        )
        test_dict = {
            "dataset": self.num_examples["dataset"],
            "fl_round": FL_ROUND,
            "strategy": self.mode,
            "test_loss": loss,
            "test_accuracy": accuracy,
        }
        eval_list.append(test_dict)
        FL_ROUND += 1
        return (
            float(loss),
            self.num_examples["testset"],
            {"loss": loss, "accuracy": accuracy},
        )




# Start Flower client
def main() -> None:
    """Load data, start FlowerClient."""

    # Parse command line argument `partition` (type of dataset) and `mode` (type of strategy)
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=str,
        choices=["HBCDs"],
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fedbn", "fedavg"],
        required=True,
        default="fedbn",
    )
    args = parser.parse_args()

    # Load model
    model = CNNModel().to(DEVICE).train()

    parser = argparse.ArgumentParser(description="Launches FL clients.")
    parser.add_argument('-cid',"--cid", type=int, default=0, help="Define Client_ID",)
    parser.add_argument('-server',"--server", default="0.0.0.0", help="Server Address",)
    parser.add_argument('-port',"--port", default="8000", help="Server Port",)
    parser.add_argument('-data', "--data", default="./data", help="Dataset source path")
    args = vars(parser.parse_args())
    cid = args['cid']
    server = args['server']
    port = args['port']
    datapath = args['data']
    net = CNNModel.to(DEVICE)


    # Load data
    trainloader, testloader, num_examples = load_datasets(args.partition)

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = FlowerClient(model, trainloader, testloader, num_examples, args.mode)
    print("Start client of dataset", num_examples["dataset"])
    print(f"Subscribing to FL server {server} on port {port}...")
    fl.client.start_numpy_client(server_address="[::]:8000", client=client)
    # Save train and evaluation loss and accuracy in json file
    with open(
        f"results/{args.partition}_{args.mode}_results.json", mode="r+"
    ) as eval_file:
        json.dump(eval_list, eval_file)


if __name__ == "__main__":
    main()