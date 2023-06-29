import torch
from torchvision import transforms
from PIL import Image

def predict(model_path: str, image_path: str) -> str:
    # Define data transformations for test set
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the image
    image = Image.open(image_path)

    # Prepare the image for prediction
    input_image = data_transforms(image).unsqueeze(0)

    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(probabilities, dim=1)

    # Get the predicted class label
    class_labels = ["MALIGNANT", "BENIGN"]
    predicted_label = class_labels[predicted_idx.item()]

    return predicted_label
 

if __name__ == "__main__":
    # Specify the path to the trained model and the input image
    model_path = "path/to/trained_model.pt"
    image_path = "path/to/input_image.jpg"

    # Perform prediction
    prediction = predict(model_path, image_path)
    print("Predicted class:", prediction)
