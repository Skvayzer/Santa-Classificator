import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import os
import pandas as pd

from sklearn.model_selection import train_test_split

import time
from tqdm import tqdm
import copy
from torch.optim import lr_scheduler

def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value





def init_model(device):

    model = torchvision.models.efficientnet_b7(pretrained=True)
#     set_requires_grad(model, False)
#     print(dir(model))
#     print(model)
#     classifier_name, old_classifier = model._modules.popitem()
    for param in model.parameters():
        param.requires_grad = False
#     classifier_input_size = old_classifier.in_features
#     hidden_layer_size = 5
#     classifier = nn.Sequential(OrderedDict([
#                            ('fc1', nn.Linear(classifier_input_size, hidden_layer_size)),
#                            ('activation', nn.SELU()),
#                            ('dropout', nn.Dropout(p=0.5)),
#                            ('fc2', nn.Linear(hidden_layer_size, 3)),
#                            ('output', nn.LogSoftmax(dim=1))
#                            ]))
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=3, bias=True)
    model = model.to(device)
    return model

def load_model(device, path):
    model = torchvision.models.resnet152(pretrained=False)
    set_requires_grad(model, False)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    return model


class DedyDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):

        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["class_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target


# hardcode
MODEL_WEIGHTS = "./baseline.pt"
TRAIN_DATASET = "../input/d/skvayzer/dedushki/dedushki/train"
TRAIN_CSV = "../input/d/skvayzer/dedushki/dedushki.csv"
MODEL_LOAD_PATH = "../input/modelsanta/efficientnet_b6.pt"

img_size = 224
# make slight augmentation and normalization on ImageNet statistics
trans = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=.5, hue=.3),
#     transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#     transforms.RandomRotation(degrees=(0, 180)),
#     transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
#     transforms.RandomPosterize(bits=2),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dset = DedyDataset(TRAIN_DATASET, TRAIN_CSV, trans)
labels = dset.targets
indices = list(range(len(labels)))
ind_train, ind_test, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=139, stratify=labels)

trainset = torch.utils.data.Subset(dset, ind_train)
testset = torch.utils.data.Subset(dset, ind_test)

batch_size = 16
num_workers = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)

loaders = {'train': trainloader, 'val': testloader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(device, MODEL_LOAD_PATH)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
def train_model(model, dataloaders, criterion, optimizer, scheduler,
                phases, num_epochs=3):
    global best_model_wts
    global best_acc

    start_time = time.time()



    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    outputs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
#                     print(preds)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, acc_history

criterion = nn.CrossEntropyLoss()


#     pretrain_optimizer = torch.optim.SGD(model.classifier[3].parameters(),
#                                         lr=0.001, momentum=0.9)
pretrain_optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.001, momentum=0.9)

train_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# # Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler_train = lr_scheduler.StepLR(train_optimizer, step_size=30, gamma=0.1)


# Train
# запустить дообучение модели
set_requires_grad(model, True)
train_results = train_model(model, loaders, criterion, train_optimizer, exp_lr_scheduler_train,
            phases=['train', 'val'], num_epochs=100)

train_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# # Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler_train = lr_scheduler.StepLR(train_optimizer, step_size=30, gamma=0.1)


# Train
# запустить дообучение модели
train_results = train_model(model, loaders, criterion, train_optimizer, exp_lr_scheduler_train,
            phases=['train', 'val'], num_epochs=50)
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_WEIGHTS)
