import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.model import CNNModel

def predict(model_path, data_dir):
    # Define data transformations for test set
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load test dataset
    testset = datasets.ImageFolder(data_dir, transform=data_transforms)

    # Create data loader for test set
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Load the latest model
    model = CNNModel
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []

    # Perform prediction on test set
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, dim=1)
            predictions.append(predicted.item())

    return predictions

if __name__ == "__main__":
    predict = predict(model_path, data_dir)