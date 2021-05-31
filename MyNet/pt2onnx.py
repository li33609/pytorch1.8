import torch
import torch.nn as nn
import torch.onnx
import onnx
from torchvision.models import squeezenet1_1
from onnxsim import simplify

pt_model_path = './model-143.pth'
onnx_model_path = './model-143.onnx'
model = squeezenet1_1()
model.classifier[1] = nn.Conv2d(512, 15, kernel_size=(1, 1), stride=1)
model.load_state_dict(torch.load(pt_model_path, map_location=torch.device('cpu')))
input_tensor = torch.randn(1, 3, 224, 224)
input_names = ['input']
output_names = ['output']
torch.onnx.export(model,
                  input_tensor,
                  onnx_model_path,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

