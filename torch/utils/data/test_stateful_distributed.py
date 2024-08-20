
import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.data = torch.arange(num_samples * num_features).view(num_samples, num_features)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(100,10)


from torch.utils.data.stateful_distributed import StatefulDistributedSampler

dataloader = StatefulDistributedSampler(dataset, rank=0, num_replicas=1, shuffle=False, seed=0, drop_last=False)


print("---------- First 10 samples ----------")
for i, batch in enumerate(dataloader):
  print(i, batch)
  if i==10:
    
    break
print("-------------------------------------")

dataloader = StatefulDistributedSampler(dataset, rank=0, num_replicas=1, shuffle=False, seed=0, drop_last=False)

print("---------- First 5 samples ----------")
for i, batch in enumerate(dataloader):
  print(i, batch)
  if i==5:
    state_dict = dataloader.state_dict()
    break
print(state_dict)
dataloader.load_state_dict(state_dict)
print("---------- Next 5 samples ----------")
for i, batch in enumerate(dataloader):
  print(i, batch)
  if i==5:
    break
