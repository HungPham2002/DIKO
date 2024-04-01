import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from dataset import DikoDataset
from model import DIKO

HEIGHT = 244
WIDTH = 244
BATCH_SIZE = 1

device = torch.device('gpu' if torch.cuda_available() else 'cpu')
print('Torch device:', device)

# Prepare test set
test_path = './OAI_Xray/test_v2.csv'
test_transforms_incept = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.6078, 0.6078, 0.6078), (0.1932, 0.1932, 0.1932))
])

test_transforms_dense = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.6078, 0.6078, 0.6078), (0.1932, 0.1932, 0.1932))
])

test_dataset = DikoDataset(test_path, transforms_1=test_transforms_incept, transforms_2=test_transforms_dense)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

# Load Model
model = DIKO()
model.to(device)
model.load_state_dict(torch.load('./weights/cls/diko.pt'))

y_pred = []
y_test = []
test_correct = 0.0
model.eval()
with torch.no_grad():
    for inputs1, inputs2, labels in tqdm(test_loader):

        inputs1  = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels  = labels.to(device)

        outputs = model(inputs1, inputs2)

        y_test.append(labels.cpu())
        y_pred.append((torch.round(outputs)).cpu())
     
    for pred, label in zip(y_pred, y_test):
        if(pred==label):
            test_correct += 1

test_acc = (test_correct * 100) / len(test_dataset)
print('Accuracy on Test set: ', test_acc)

flattened_y_pred = []
for tensor in y_pred:
    for element in tensor:
        flattened_y_pred.append(element.cpu().item())
flattened_y_pred = np.array(flattened_y_pred)

flattened_y_test = []
for tensor in y_test:
    for element in tensor:
        flattened_y_test.append(element.cpu().item())
flattened_y_test = np.array(flattened_y_test)

# Print Classification Report
print(classification_report(flattened_y_test, flattened_y_pred, labels=[0, 1]))

# Plot Confusion Matrix
cm = confusion_matrix(flattened_y_test, flattened_y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap = 'Blues')
plt.show()

