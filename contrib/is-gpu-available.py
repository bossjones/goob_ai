#!/usr/bin/env python

import torch  # type: ignore

cuda_available = torch.cuda.is_available()
current_device = torch.cuda.current_device()

print(torch.cuda.device(current_device))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(current_device))
