package_path = "./input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master/"
import sys
sys.path.append(package_path)

import os
import glob
import time
import random

import numpy as np
import pandas as pd

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils import data as torch_data
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import efficientnet_pytorch
import torchvision.models as models
from torch.utils import model_zoo
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 123


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed)


class CFG:
    img_size = 256
    n_frames = 10

    cnn_features = 256
    lstm_hidden = 32

    n_fold = 5
    n_epochs = 100


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        #         self.net = efficientnet_pytorch.EfficientNet.from_name("efficientnet-b0")
        #         checkpoint = torch.load("./input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        #         self.net.load_state_dict(checkpoint)

        #         self.net = models.resnet34()
        #         checkpoint = torch.load("/root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth")
        #         self.net.load_state_dict(checkpoint)

        ########
        self.net = models.resnet34(pretrained=False)
        ###########
        n_features = self.net.fc.in_features
        # import pdb;pdb.set_trace()
        self.net.fc = nn.Linear(in_features=n_features, out_features=CFG.cnn_features, bias=True)

    def forward(self, x):
        x = F.relu(self.map(x))
        out = self.net(x)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(CFG.cnn_features, CFG.lstm_hidden, 2, batch_first=True)
        self.fc = nn.Linear(CFG.lstm_hidden, 1, bias=True)

    def forward(self, x):
        # x shape: BxTxCxHxW
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        import pdb;
        pdb.set_trace()
        r_in = c_out.view(batch_size, timesteps, -1)
        output, (hn, cn) = self.rnn(r_in)

        out = self.fc(hn[-1])
        return out


def load_image(path):
    image = cv2.imread(path, 0)
    if image is None:
        return np.zeros((CFG.img_size, CFG.img_size))

    image = cv2.resize(image, (CFG.img_size, CFG.img_size)) / 255
    return image.astype('f')


def uniform_temporal_subsample(x, num_samples):
    '''
        Moddified from https://github.com/facebookresearch/pytorchvideo/blob/d7874f788bc00a7badfb4310a912f6e531ffd6d3/pytorchvideo/transforms/functional.py#L19
        Args:
            x: input list
            num_samples: The number of equispaced samples to be selected
        Returns:
            Output list
    '''
    t = len(x)
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return [x[i] for i in indices]


class DataRetriever(Dataset):
    def __init__(self, paths, targets, transform=None):
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def read_video(self, vid_paths):
        video = [load_image(path) for path in vid_paths]
        #  import pdb;pdb.set_trace()
        if self.transform:
            seed = random.randint(0, 99999)
            for i in range(len(video)):
                random.seed(seed)
                video[i] = self.transform(image=video[i])["image"]

        video = [torch.tensor(frame, dtype=torch.float32) for frame in video]
        if len(video) == 0:
            video = torch.zeros(CFG.n_frames, CFG.img_size, CFG.img_size)
        else:
            video = torch.stack(video)  # T * C * H * W
        #         video = torch.transpose(video, 0, 1) # C * T * H * W
        return video

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = f"./input/rsna-miccai-png/train/{str(_id).zfill(5)}/"
        channels = []
        for t in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            num_samples = CFG.n_frames
            if len(t_paths) < num_samples:
                in_frames_path = t_paths
            else:
                in_frames_path = uniform_temporal_subsample(t_paths, num_samples)

            channel = self.read_video(in_frames_path)
            if channel.shape[0] == 0:
                print("1 channel empty")
                channel = torch.zeros(num_samples, CFG.img_size, CFG.img_size)
            channels.append(channel)

        channels = torch.stack(channels).transpose(0, 1)

        y = torch.tensor(self.targets[index], dtype=torch.float)
        return {"X": channels.float(), "y": y}

df = pd.read_csv("./input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv")

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(
                                    shift_limit=0.0625,
                                    scale_limit=0.1,
                                    rotate_limit=10,
                                    p=0.5
                                ),
                                A.RandomBrightnessContrast(p=0.5),
                            ])
valid_transform = A.Compose([
                            ])


class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        # incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion,
            loss_meter,
            score_meter
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter
        self.hist = {'val_loss': [],
                     'val_score': [],
                     'train_loss': [],
                     'train_score': []
                     }

        self.best_valid_score = -np.inf
        self.best_valid_loss = np.inf
        self.n_patience = 0

        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
            "patience": "\nValid score didn't improve last {} epochs."
        }

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_score, train_time = self.train_epoch(train_loader)
            valid_loss, valid_score, valid_time = self.valid_epoch(valid_loader)
            self.hist['val_loss'].append(valid_loss)
            self.hist['train_loss'].append(train_loss)
            self.hist['val_score'].append(valid_score)
            self.hist['train_score'].append(train_score)

            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_time
            )

            self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_time
            )

            if self.best_valid_score < valid_score:
                self.info_message(
                    self.messages["checkpoint"], self.best_valid_score, valid_score, save_path
                )
                self.best_valid_score = valid_score
                self.best_valid_loss = valid_loss
                self.save_model(n_epoch, save_path)
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break

        return self.best_valid_loss, self.best_valid_score

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())

            self.optimizer.step()

            _loss, _score = train_loss.avg, train_score.avg
            message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}'
            self.info_message(message, step, len(train_loader), _loss, _score, end="\r")

        return train_loss.avg, train_score.avg, int(time.time() - t)

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)

            _loss, _score = valid_loss.avg, valid_score.avg
            message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}'
            self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")

        return valid_loss.avg, valid_score.avg, int(time.time() - t)

    def plot_loss(self):
        plt.title("Loss")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")

        plt.plot(self.hist['train_loss'], label="Train")
        plt.plot(self.hist['val_loss'], label="Validation")
        plt.legend()
        plt.show()

    def plot_score(self):
        plt.title("Score")
        plt.xlabel("Training Epochs")
        plt.ylabel("Acc")

        plt.plot(self.hist['train_score'], label="Train")
        plt.plot(self.hist['val_score'], label="Validation")
        plt.legend()
        plt.show()

    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )


    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


skf = StratifiedKFold(n_splits=CFG.n_fold)
t = df['MGMT_value']

start_time = time.time()

losses = []
scores = []

for fold, (train_index, val_index) in enumerate(skf.split(np.zeros(len(t)), t), 1):
    print('-' * 30)
    print(f"Fold {fold}")

    train_df = df.loc[train_index]
    val_df = df.loc[val_index]
    train_retriever = DataRetriever(
        train_df["BraTS21ID"].values,
        train_df["MGMT_value"].values,
        train_transform
    )
    val_retriever = DataRetriever(
        val_df["BraTS21ID"].values,
        val_df["MGMT_value"].values
    )
    train_loader = torch_data.DataLoader(
        train_retriever,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = torch_data.DataLoader(
        val_retriever,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    model = Model()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = F.binary_cross_entropy_with_logits

    trainer = Trainer(
        model,
        device,
        optimizer,
        criterion,
        LossMeter,
        AccMeter
    )
    loss, score = trainer.fit(
        CFG.n_epochs,
        train_loader,
        valid_loader,
        f"best-model-{fold}.pth",
        100,
    )
    losses.append(loss)
    scores.append(score)

    trainer.plot_loss()
    trainer.plot_score()

elapsed_time = time.time() - start_time
print('\nTraining complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
print('Avg loss {}'.format(np.mean(losses)))
print('Avg score {}'.format(np.mean(scores)))