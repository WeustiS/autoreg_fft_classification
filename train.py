import torch
from torch import nn

from model import Model

from tqdm import tqdm
from config import CONFIG

from TinyFFTImageNet import TinyFFTImageNet
from utils import seed_everything, accuracy

if CONFIG["wandb"]:
    import wandb
    wandb.init(project="cs598", config=CONFIG)
seed_everything(42)
          
dims = [1, 2, 3, 6, 11, 12, 24, 33, 48, 64, 66, 96, 132, 192, 264, 352, 528, 704, 1056, 2112]
assert CONFIG['dim'] in dims
train_dataset = TinyFFTImageNet(r"/raid/projects/weustis/data/tiny-imagenet-200/train", CONFIG['dim'], norm="L1")
test_dataset = TinyFFTImageNet(r"/raid/projects/weustis/data/tiny-imagenet-200/test", CONFIG['dim'], norm="L1", train=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

model = Model(CONFIG['dim']*3*4, CONFIG['n_classes'], train_dataset.max_n_tok)

# Loss/Optimizer
crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters())

for epoch in range(CONFIG['epochs']):
    # train loop
    model.train()
    train_loss_total = 0
    train_top1_total = 0
    train_top5_total = 0
    
    test_loss_total = 0
    test_top1_total = 0
    test_top5_total = 0
    
    for x,y in tqdm(train_dataloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
        loss = crit(pred, y)
        train_loss_total += loss.item()/len(x)

        top1, top5 = accuracy(pred, y)
        train_top1_total += top1.item() * len(x)
        train_top5_total += top5.item() * len(x)
        
        loss.backward()
        opt.step()
       
    model.eval()
    with torch.no_grad():
        
        for x,y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = crit(pred, y)
            test_loss_total += loss.item()/len(x)
            top1, top5 = accuracy(pred, y)
            test_top1_total += top1.item() * len(x)
            test_top5_total += top5.item() * len(x)
    wandb.log({
        "train_loss": train_loss_total,
        "train_top1": train_top1_total/len(train_dataset),
        "train_top5": train_top5_total/len(train_dataset),
        "test_loss": test_loss_total,
        "test_top1": test_top1_total/len(test_dataset),
        "test_top5": test_top5_total/len(test_dataset)
    })
    
torch.save(model.state_dict(), f"model_{wandb.run.id}.pt")
