from torchvision import models
import torch.nn as nn

# Function to configure the model architecture based on the provided name
def model_config(model_name='efficientnet'):
    model = {
        'densenet': models.densenet121(pretrained=True),
        'efficientnet': models.efficientnet_b0(weights='DEFAULT')
    }
    return model[model_name]

# Function to build and configure the model architecture
def build_model(model_name='efficientnet', fine_tune=True, num_classes=10):
    model = model_config(model_name) # Get the specified model architecture
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True # Set all model parameters to be trainable
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...') # Freeze hidden layers, keeping only the last layers trainable
        for params in model.parameters():
            params.requires_grad = False

    # Adjust final fully connected layers based on the chosen model
    if model_name == 'densenet':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes) # Change the last fully connected layer for DenseNet
    if model_name == 'efficientnet':
        model.classifier[1].out_features = num_classes # Change the number of output classes for EfficientNet
    return model # Return the configured model
