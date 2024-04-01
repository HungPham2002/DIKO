import torch
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from tqdm import tqdm

from dataset import DikoDataset
from model import DIKO

HEIGHT = 244
WIDTH = 244
NUM_CLASSES = 2
BATCH_SIZE = 32

train_path = './OAI_Xray/train_v2.csv'
val_path = './OAI_Xray/val_v2.csv'

train_transforms_incept = transforms.Compose([
    transforms.Resize((299 , 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(22.5),
    transforms.ColorJitter(brightness = (0.7, 1.0)),
    v2.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize((0.6078, 0.6078, 0.6078), (0.1932, 0.1932, 0.1932))
])

train_transforms_dense = transforms.Compose([
    transforms.Resize((HEIGHT , WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(22.5),
    transforms.ColorJitter(brightness = (0.7, 1.0)),
    v2.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize((0.6078, 0.6078, 0.6078), (0.1932, 0.1932, 0.1932))
])

val_transforms_incept = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.6078, 0.6078, 0.6078), (0.1932, 0.1932, 0.1932))
])

val_transforms_dense = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.6078, 0.6078, 0.6078), (0.1932, 0.1932, 0.1932))
])


train_dataset = DikoDataset(train_path, transforms_1=train_transforms_incept, transforms_2=train_transforms_dense)
val_dataset = DikoDataset(val_path, transforms_1=val_transforms_incept, transforms_2=val_transforms_dense)


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)


device = torch.device('cuda' if torch.cuda_available() else 'cpu')
print('Torch device:', device)

# Model Init
model = DIKO()
model.to(device)

criterion  = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                               mode='max',
                                               factor=0.5,
                                               patience=5,
                                               threshold=1e-3,
                                               threshold_mode='abs',
                                               verbose = True)
val_acc_max = 0.0

num_epochs = 30
loss_epochs = []
loss_val_epochs = []
acc_epochs = []
acc_eval_epochs = []

for epoch in range(num_epochs):
    # Training model 
    num_acc = 0.0
    running_loss = 0.0
    model.train()
    for inputs1, inputs2, labels in tqdm(train_loader):
        optimizer.zero_grad()
        
        inputs1  = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels  = labels.to(device)
        
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs1.size(0)
        predicted = torch.round(outputs)
        for pred, label in zip(predicted, labels):
            if(pred==label):
                num_acc += 1
    epoch_loss = running_loss / len(train_loader)
    acc = num_acc / len(train_dataset)

    loss_epochs.append(epoch_loss)
    acc_epochs.append(acc)
    
    # Evaluate model     
    num_val_acc = 0.0
    running_val_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for inputs1, inputs2, labels in tqdm(val_loader):

            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels  = labels.to(device)

            outputs = model(inputs1, inputs2)

            loss = criterion(outputs.squeeze(1), labels.float())

            running_val_loss += loss.item() * inputs1.size(0)
            predicted = torch.round(outputs)
            for pred, label in zip(predicted, labels):
                if(pred==label):
                    num_val_acc += 1
    
    val_loss = running_val_loss / len(val_loader)
    val_acc = num_val_acc / len(val_dataset)    

    loss_val_epochs.append(val_loss)
    acc_eval_epochs.append(val_acc)
    
    scheduler.step(val_acc)
    
    if val_acc > val_acc_max:
        print('Validation acc increased ({:.6f} --> {:.6f}).   Saving model ...'.format(val_acc_max, val_acc))
        torch.save(model.state_dict() ,'./weights/diko.pt')
        val_acc_max = val_acc
        
    print("Epoch", epoch + 1)
    print(f"Loss: {epoch_loss:.4f}, Train_acc: {acc*100:.2f}")  
    print(f"Val_loss: {val_loss:.4f}, Val_acc: {val_acc*100:.2f}") 