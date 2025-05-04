import torch
import torch.onnx
from ispnet import LiteISPNet

model = LiteISPNet()
device = 'cpu'
model.load_state_dict(torch.load('trained_models/ispnet_model.pth')['state_dict'])
model.to(device).eval()

# Set up a dummy input with dynamic dimensions
# Use symbolic names for dynamic dimensions
batch_size = 1
dummy_input = (torch.randn(batch_size, 4, 256, 256, device=device), torch.randn(batch_size, 2, 256, 256, device=device))

# Export the model to ONNX format
torch.onnx.export(
    model,                                      # PyTorch model
    dummy_input,                                # Example input
    "ispnet_model.onnx",                        # Output file
    export_params=True,                         # Store the trained weights
    opset_version=12,                           # ONNX version to use
    do_constant_folding=True,                   # Optimization: fold constants
    input_names=["input"],                      # Input name
    output_names=["output"],                    # Output name
    dynamic_axes={                              # Specify dynamic axes
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)

print("Model has been converted to ONNX format with dynamic dimensions.")