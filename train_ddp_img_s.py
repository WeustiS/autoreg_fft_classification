import torch
from torch import nn

from vit_pytorch import ViT
from torchvision import transforms
import torchvision
from tqdm import tqdm
from config import CONFIG_f as CONFIG

import os

from TinyTinyFFTImageNet import TinyFFTImageNet
from utils import seed_everything, accuracy

import sys
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
seed_everything(42)

import wandb

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Starting DDP training on rank {rank}.")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


    seed_everything(42)
    from config import CONFIG_f as CONFIG
    if CONFIG["wandb"] and rank == 0:
        wandb.init(project="cs598", config=CONFIG)
                    
        CONFIG = wandb.config

        print(CONFIG)

    # dims = [1, 2, 3, 6, 11, 12, 24, 33, 48, 64, 66, 96, 132, 192, 264, 352, 528, 704, 1056, 2112]
    # assert CONFIG['dim'] in dims
    pre_process_train = transforms.Compose([
                # transforms.Resize(32),
                transforms.Resize(74),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(CONFIG['randaug_n-op'], CONFIG['randaug_mag']),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    pre_process_test = transforms.Compose([
                # transforms.Resize(32),
                transforms.Resize(74),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    #train_dataset = TinyFFTImageNet(r"/raid/projects/weustis/data/tiny-imagenet-200/train", CONFIG['n_tokens'], CONFIG['dim'], CONFIG['patch_size'], norm="L1")
    #test_dataset = TinyFFTImageNet(r"/raid/projects/weustis/data/tiny-imagenet-200/test", CONFIG['n_tokens'], CONFIG['dim'], CONFIG['patch_size'], norm="L1", train=False)
    dataset = torchvision.datasets.ImageFolder(r"/raid/projects/weustis/data/tiny-imagenet-200/train")
    # test_dataset = torchvision.datasets.ImageFolder(r"/raid/projects/weustis/data/tiny-imagenet-200/test", transform=pre_process_test)
    l = len(dataset)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(l*.9), l - int(l*.9)], generator=torch.Generator().manual_seed(42))

    train_dataset = TinyFFTImageNet(train_dataset, transforms=pre_process_train)
    test_dataset = TinyFFTImageNet(test_dataset, transforms=pre_process_test)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # 
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=(sampler is None), num_workers=CONFIG['num_workers'], pin_memory=True, sampler=sampler)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)

    # model = ViT( # LARGE
    #     image_size=(64, 33),
    #     patch_size=CONFIG['patch_size'],
    #     dim=512,
    #     depth=8,
    #     heads=10,
    #     mlp_dim=2048,
    #     num_classes=200,
    #     channels=12
    # ).cuda()
    patch_size = (CONFIG["patch_h"], CONFIG["patch_w"])
    patch_size = (8,8)
    image_size = (64, 64)
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=CONFIG['hidden_dim'], # 192
        depth=CONFIG['depth'],
        heads=CONFIG['heads'],
        mlp_dim=CONFIG['hidden_dim']*4,
        num_classes=200,
        channels=3,
        dropout=CONFIG['dropout']
    ).to(rank)
    model = DDP(model, device_ids=[rank])
    print("Params:",  sum(p.numel() for p in model.parameters() if p.requires_grad))

    n_tokens = int((image_size[0]//patch_size[0] * image_size[1]//patch_size[1])*CONFIG['frac_tokens'])
    print("N_TOK:", n_tokens)
    # Loss/Optimizer
    crit = torch.nn.CrossEntropyLoss()
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
         weight_decay=CONFIG['weight_decay'])

    for epoch in range(CONFIG['epochs']):
        train_loss_total = 0
        train_top1_total = 0
        train_top5_total = 0
            
        test_loss_total = 0
        test_top1_total = 0
        test_top5_total = 0
        sampler.set_epoch(epoch)
        # train loop
        model.train()

        if epoch % 50 == 49 and rank == 0:
            torch.save(model.state_dict(), f"model_e{epoch}_{wandb.run.id}.pt")
        for x,y in tqdm(train_dataloader):
            opt.zero_grad()
            x = x.to(rank)
            y = y.to(rank)
            #pred = model(x)
            # n_tok = torch.randint(low=3, high=n_tokens, size=(1,)).item()

            pred = model(x)
            loss = crit(pred, y)

            train_loss_total += loss.item()

            top1, top5 = accuracy(pred, y)
            train_top1_total += top1.item() * len(x)
            train_top5_total += top5.item() * len(x)
            
            loss.backward()
            opt.step()

        
        if rank == 0 and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for x,y in tqdm(test_dataloader):
                    x = x.to(rank)
                    y = y.to(rank)
                    #pred = model(x)
                    pred = model(x)
                    loss = crit(pred, y)
                    test_loss_total += loss.item()
                    top1, top5 = accuracy(pred, y)
                    test_top1_total += top1.item() * len(x)
                    test_top5_total += top5.item() * len(x)
            for i in range(5):
                wandb.log({
                    "train_loss": train_loss_total/len(train_dataloader),
                    "train_top1": train_top1_total/len(train_dataset)* torch.cuda.device_count(),
                    "train_top5": train_top5_total/len(train_dataset)* torch.cuda.device_count(),
                    "test_loss": test_loss_total/len(test_dataloader),
                    "test_top1": test_top1_total/len(test_dataset),
                    "test_top5": test_top5_total/len(test_dataset)
                })

    if rank == 0:    
        torch.save(model.state_dict(), f"model_{wandb.run.id}.pt")

    cleanup()


def main():
    print("found", torch.cuda.device_count(), "gpus!")
    world_size = torch.cuda.device_count()

    mp.spawn(train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__=="__main__":

    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.

    main()