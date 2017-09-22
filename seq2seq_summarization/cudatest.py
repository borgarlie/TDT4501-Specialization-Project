import torch

x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
print(torch.backends.cudnn.version())