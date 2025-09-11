import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
print(torch.version.cuda)
