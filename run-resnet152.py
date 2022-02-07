import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import os
import pandas as pd



def load_model(device, path):
    model = torchvision.models.resnet152(pretrained=False)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
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
        filename = image.filename.split("/")[-1]

        if self.transform:
            image = self.transform(image)
        return image, filename


# hardcode
MODEL_WEIGHTS = "./data/weights/resnet152.pt"
TEST_DATASET = "./data/test"

if __name__ == "__main__":
    img_size = 224
    # make slight augmentation and normalization on ImageNet statistics
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dset = DedyDataset(TEST_DATASET, transform=transform)
    labels = dset.targets


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device, path=MODEL_WEIGHTS)
    images = []
    labels = []
    for image, filename in dset:
        image = image.to(device)
        prediction = model(image.unsqueeze(0))
        _, pred = torch.max(prediction, 1)
        images.append(filename)
        labels.append(pred[0].item())
    submission = pd.DataFrame({ 'image_name': images, 'class_id': labels })
    submission.to_csv("./data/out/submission.csv", index=False, sep="\t")
