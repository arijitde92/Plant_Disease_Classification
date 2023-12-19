import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib

from model import build_model  # Import the model building function
from class_names import class_names as CLASS_NAMES  # Import class names from class_names.py

# Construct the argument parser to accept weights and image path arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', default='../outputs/efficientnet_80_20/best_model.pth',
                    help='path to the model weights')
parser.add_argument('-i', '--image', default='../input/inference_data/apple_scab.jpg', help='path to the model weights')
args = vars(parser.parse_args())

# Constants and other configurations.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check available device (GPU or CPU)
IMAGE_RESIZE = 224  # Image resize dimension


# Function to create transform for test images
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform


# Function to denormalize an image tensor
def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)


# Function to annotate image with predicted class labels
def annotate_image(image, output_class):
    # Denormalize the image tensor
    image = denormalize(image).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_name = CLASS_NAMES[int(output_class)]
    plant = class_name.split('___')[0]  # Extract plant name from class name
    disease = class_name.split('___')[-1]  # Extract disease name from class name
    # Add text annotations on the image
    cv2.putText(
        image,
        f"{plant}",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA
    )
    cv2.putText(
        image,
        f"{disease}",
        (5, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA
    )
    return image


# Function for inference using the model
def inference(model, testloader, DEVICE):
    model.eval()
    with torch.no_grad():
        image = testloader
        image = image.to(DEVICE)
        outputs = model(image)
    predictions = F.softmax(outputs, dim=1).cpu().numpy()  # Softmax probabilities
    output_class = np.argmax(predictions)  # Predicted class number
    result = annotate_image(image, output_class)
    return result


if __name__ == '__main__':
    weights_path = pathlib.Path(args['weights'])
    model_name = str(weights_path).split(os.path.sep)[-2]
    infer_result_path = os.path.join('..', 'outputs', 'inference_results', model_name)
    os.makedirs(infer_result_path, exist_ok=True)
    img_path = args['image']
    checkpoint = torch.load(weights_path)
    model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = get_test_transform(IMAGE_RESIZE)

    # For inferencing a single image
    print(f"Inference on image located at: {img_path}")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    result = inference(model, image, DEVICE)

    image_name = img_path.split(os.sep)[-1]
    cv2.imshow('Image', result)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(infer_result_path, image_name), result * 255.)
