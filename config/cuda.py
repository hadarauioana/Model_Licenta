import torch

print("PyTorch CUDA Available:", torch.cuda.is_available())  # Should be True
print("CUDA Version in PyTorch:", torch.version.cuda)  # Should match 12.1
print("cuDNN Version:", torch.backends.cudnn.version())  # Should print a valid version
print("Device Count:", torch.cuda.device_count())  # Should return 1 or more
print("Current Device Name:", torch.cuda.get_device_name(0))  # Should return your GPU name
