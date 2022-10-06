import os
import torch
import resnet_init
import model_pruning
import pickle

# Load the trained resnet model as feature extractor
device = torch.device("cpu")
resnet_init = resnet_init.model
weights = r"model_resnet.pth"
resnet_init.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))

# List the layers in our net
children_counter = 0
for n, c in resnet_init.named_children():
    print(
        "Children Counter: ",
        children_counter,
        " Layer Name: ",
        n,
    )
    children_counter += 1
layers = list(resnet_init._modules.keys())

# Pop the linear layers
model_new = resnet_init._modules.pop(layers[0])

# Pop the last layer from original resnet
layers_new = list(model_new._modules.keys())
model_last = model_pruning.new_model(model_new, "avgpool")
model_last.eval()

# Load the ML model
with open("model_svm.pkl", "rb") as f:
    cnn_based_ml = pickle.load(f)
