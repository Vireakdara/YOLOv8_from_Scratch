import yaml
from utils.dataset import *
from torch.utils import data

# get all file names
data_dir= 'COCO'
filenames_train = []
with open(f'{data_dir}/train2017.txt') as reader:
    for filename in reader.readlines():
        filename = os.path.basename(filename.rstrip())
        filenames_train.append(f'{data_dir}/images/train2017/' + filename)

# input_size for the model
input_size=640

# get params from yaml file
with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

train_data=Dataset(filenames_train,input_size,params,augment=True)
train_loader = data.DataLoader(train_data, batch_size=64, num_workers=0, pin_memory=True, collate_fn=Dataset.collate_fn)
print(f"Train_loader : {len(train_loader)} batches")

batch=next(iter(train_loader))
print("All keys in batch      : ", batch[1].keys())
print(f"Input batch shape      : ", batch[0].shape)
print(f"Classification scores  : {batch[1]['cls'].shape}")
print(f"Box coordinates        : {batch[1]['box'].shape}")
print(f"Index identifier (which score belongs to which image): {batch[1]['idx'].shape}")

torch.manual_seed(1337)

# model, loss and optimizer
model=MyYolo(version='n')
print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")

criterion=util.ComputeLoss(model, params)
optimizer=torch.optim.AdamW(model.parameters(), lr=0.5)

num_epochs=5

imgs,targets=batch[0],batch[1]
imgs=imgs.float()
model.train()
for epoch in range(num_epochs):
    outputs=model(imgs)
    loss=sum(criterion(outputs,targets)) # cls_loss+box_loss+dfl_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch : {epoch} | loss : {loss.item()}")
