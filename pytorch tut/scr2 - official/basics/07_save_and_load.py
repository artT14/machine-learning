import torch
import torchvision.models as models

# SAVE MODEL
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# LOAD MODEL
model = models.vgg16() # Need to create instance of same model
torch.save(model.state_dict(), 'model_weights.pth')
model.eval()

# SAVING & LOADING MODELS W/ SHAPES
torch.save(model, 'model.pth')
model = torch.load('model.pth')