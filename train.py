import gc
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset import *
from loss import *
from model import *

def train_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    
    running_loss = 0.
    model.train()
    for batch_idx, batch in enumerate(loader):
        # 参考
        # https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py#L209
        images = torch.cat([batch["image"][0], batch["image"][1]], dim=0)
        images = images.to(DEVICE)
        target = batch["target"].to(DEVICE)
        bsz = target.shape[0]

        optimizer.zero_grad()

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 15 == 14:
            print(f'epoch{epoch}, batch{batch_idx+1} loss: {running_loss / 15}')
            train_loss = running_loss / 15
            running_loss = 0.

    gc.collect()    
    return train_loss

if __name__ == '__main__':

    #パラメーター
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 10
    BATCH = 32

    dic = make_datapath_dic("train")
    transform = ImageTransform(300)
    train_dataset = SupConDataset(dic, transform=transform, phase="train")

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    
    model = SupConModel(base_name="resnet18",pretrained=True,feat_dim=128)
    model = model.to(DEVICE)

    criterion = SupConLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-03, weight_decay=1.0e-02)
    scheduler = lr_scheduler.OneCycleLR(optimizer, epochs=EPOCHS, steps_per_epoch=len(train_loader),
                                        max_lr=1.0e-3, pct_start=0.1, anneal_strategy='cos',
                                        div_factor=1.0e+3, final_div_factor=1.0e+3
                                        )
    x_epoch_data = []
    y_train_loss_data = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
        x_epoch_data.append(epoch)
        y_train_loss_data.append(train_loss)
        scheduler.step()
        
    plt.plot(x_epoch_data, y_train_loss_data, color='blue', label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.show()
    
    figure = str(y_train_loss_data[-1]) + ".png"
    plt.savefig(figure)

    model = model.backbone
    model_name = str(y_train_loss_data[-1]) + '.pth'
    torch.save(model.state_dict(), model_name)
    print(f'Saved model as {model_name}')
