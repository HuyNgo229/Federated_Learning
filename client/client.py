import os
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from typing import Tuple, Dict, Union, Optional, List

FL_ROUND: int = 0

eval_list: List[Dict[str, Union[str, int, float]]] = []


class Client(nn.Module):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        epoch: int,
        device: str,
        model: Optional[nn.Module],
        percentage_of_dataset: float,
        dataset_name: str,
        mode: str,
        server_model_path: str,
        client_model_path: str,
    ):
        super(Client, self).__init__()
        self.batch_size = batch_size
        self.epoch = epoch
        self.device = device
        self.model = model
        self.data_dir = data_dir
        self.percentage_of_dataset = percentage_of_dataset
        self.dataset_name = dataset_name
        self.mode = mode
        self.trainloader = None
        self.validloader = None
        self.num_examples = None
        self.server_model_path = server_model_path
        self.client_model_path = client_model_path

    def load_datasets(self) -> Tuple[DataLoader, DataLoader, Dict[str, Union[str, int]]]:
        # Define data transformations for train and valid sets
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
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

        # Load train and valid datasets
        trainset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), data_transforms['train'])
        testset = datasets.ImageFolder(os.path.join(self.data_dir, 'valid'), data_transforms['valid'])

        # Calculate sizes for train and valid subsets
        train_size = int(len(trainset) * self.percentage_of_dataset)
        valid_size = int(len(testset) * self.percentage_of_dataset)

        # Randomly select train and valid indices
        train_indices = torch.randperm(len(trainset))[:train_size].tolist()
        valid_indices = torch.randperm(len(testset))[:valid_size].tolist()

        # Create samplers for train and valid subsets
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create train and valid data loaders
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler)
        self.validloader = DataLoader(testset, batch_size=self.batch_size, sampler=valid_sampler)

        train_length = len(train_sampler)
        valid_length = len(valid_sampler)
        self.num_examples = {
            'dataset': self.dataset_name,
            'trainloader': train_length,
            'validloader': valid_length
        }

        return self.trainloader, self.validloader, self.num_examples

    def set_parameters(self, server_model_path: str) -> None:
        """Load model parameters from a file."""
        state_dict = torch.load(server_model_path)
        self.model.load_state_dict(state_dict)

    def get_parameters(self, client_model_path: str) -> None:
        """Save model parameters to a file."""
        torch.save(self.model.state_dict(), client_model_path)

    def fit(self) -> Tuple[None, int, Dict[str, Union[str, int, float]]]:
        self.set_parameters(self.server_model_path)

        if self.model is not None:
            print("Model fitting done...")
        else:
            print("Cannot fit the model")

        test_loss, test_accuracy = test(
            self.model, self.trainloader, self.dataset_name, device=self.device
        )
        test_dict = {
            "dataset": self.dataset_name,
            "fl_round": FL_ROUND,
            "strategy": self.mode,
            "train_loss": test_loss,
            "train_accuracy": test_accuracy,
        }

        loss, accuracy = train(
            self.model,
            self.trainloader,
            self.dataset_name,
            epochs=self.epoch,
            device=self.device,
        )
        eval_list.append(test_dict)

        self.get_parameters(self.client_model_path)
        return (
            None,
            self.num_examples['trainloader'],
            {"loss": loss, "accuracy": accuracy},
        )

    def evaluate(self) -> Tuple[float, int, Dict[str, Union[str, int, float]]]:
        global FL_ROUND
        loss, accuracy = test(
            self.model, self.validloader, self.dataset_name, device=self.device
        )
        test_dict = {
            "dataset": self.dataset_name,
            "fl_round": FL_ROUND,
            "strategy": self.mode,
            "test_loss": loss,
            "test_accuracy": accuracy,
        }
        eval_list.append(test_dict)
        FL_ROUND += 1
        return (
            float(loss),
            self.num_examples["validloader"],
            {"loss": loss, "accuracy": accuracy},
        )


def train(
    model: nn.Module,
    traindata: DataLoader,
    dataset: str,
    epochs: int,
    device: str
) -> Tuple[float, float]:
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
        for i, data in enumerate(tqdm(traindata), 0):
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
            loss_batch = running_loss / len(traindata)
            accuracy = correct / total
            if i == len(traindata) - 1:  # print every 100 mini-batches
                print(
                    f"Train Dataset {dataset} with [Epoch {epoch+1}, {i+1} mini-batches] \
                    loss: {loss_batch} accuracy: {accuracy}"
                )
                running_loss = 0.0
        loss = loss / len(traindata)
    return loss, accuracy


def test(
    model: nn.Module,
    testloader: DataLoader,
    dataset_name: str,
    device: str
) -> Tuple[float, float]:
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    valid_total = 0.0
    valid_correct = 0
    valid_running_loss = 0.0

    with torch.no_grad():
        # Validate the model
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    test_loss = valid_running_loss / len(testloader)
    test_accuracy = valid_correct / valid_total

    print('Validation Accuracy: %.4f, Validation Loss: %.4f' % (test_accuracy, test_loss))

    return test_loss, test_accuracy
