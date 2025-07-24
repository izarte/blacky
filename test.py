import ray
import torch

ray.init()
print(ray.available_resources())  # should list "GPU": 1.0 (or more)
print(torch.cuda.is_available())
