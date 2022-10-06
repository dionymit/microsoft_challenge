import torch
import torch.nn as nn

# Create nn that has 4 simple linear layers leading to class dimension(4)
class net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Initiate the model (use cpu for better compatibility - not everyone can use cuda)
device = torch.device("cpu")
extralayers = net(1000, 200, 50, 10, 4)
resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
model = nn.Sequential(resnet, extralayers)
model.to(device)

# Load weights
weights = r"model_resnet.pth"
model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
model.eval()
