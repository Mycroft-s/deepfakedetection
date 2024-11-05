import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt

# Set the path to the directory with the trained model
model_path = "./outputs/final_model"


# Function to load the model and image processor
def load_model(model_path):
    """
    Load the model and image processor.
    :param model_path: Path to the model directory
    :return: Loaded model and image processor
    """
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    return model, processor


# Load the model and image processor
model, feature_extractor = load_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to test a single image
def test_single_image(image_path):
    """
    Test a single image and display the prediction result.
    :param image_path: Path to the test image
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = 'REAL' if predicted_class_idx == 1 else 'FAKE'

    print(f"Predicted label: {predicted_label}")

    # Display the image and predicted label
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label}")
    plt.show()


# Set the path to the test image (replace with the actual image path)
test_image_path = "F:/python project/deepfakedetection/test_real.jpg"  # Replace with the path of the image to test
test_single_image(test_image_path)
