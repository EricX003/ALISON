#Torch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import time 


from captum.attr import IntegratedGradients
import tqdm
import os
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

    def __init__(self, in_size, num_classes):
        super(Model, self).__init__()

        self.act = nn.ReLU()
        self.dp = 0.40
        self.width = 512
        self.num_classes = num_classes
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p = self.dp),
            nn.Linear(in_size, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, 128),
            self.act,
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class Loader(Dataset):
    def __init__(self, x, y):
        self.x = torch.nan_to_num(torch.Tensor(x))
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_and_eval(model, training_set, validation_set, loss_function, optimizer, scheduler, epochs=30, save_path='./', save_epoch=10):

    model.to(device)

    for epoch in range(1, epochs + 1):
        print('------------', '\n', 'Epoch #:', epoch, '\n', 'Training:')

        with torch.set_grad_enabled(True):

            total = 0
            correct = 0

            model.train()

            labels = []
            predictions = []
            all_preds = []
            all_labels = []

            with tqdm(training_set, unit = "batch", bar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}') as tqdm_train:
                for training_data, labels in tqdm_train:
                    tqdm_train.set_description(f'Epoch: {epoch}')

                    training_data = training_data.to(device)
                    labels = torch.tensor(labels, dtype = torch.long)
                    #print(labels)
                    labels = labels.to(device)
                    #print(correct_prediction)

                    optimizer.zero_grad()

                    prediction = model(training_data).squeeze(1)

                    loss = loss_function(prediction, labels)

                    loss.backward()
                    optimizer.step()

                    for idx, i in enumerate(prediction):
                        if torch.argmax(i) == labels[idx]:
                            correct += 1
                        total += 1

                    tqdm_train.set_postfix(loss=loss.item(), accuracy=100.*correct/total)
                    time.sleep(0.1)

            training_accuracy = round(correct/total, 5)
            print('Training Accuracy: ', training_accuracy)

            scheduler.step()


        print('------------', '\n', 'Validation:')
        model.eval()
        gc.collect()

        total = 0
        correct = 0

        with torch.no_grad():

            with tqdm(validation_set, unit="batch", bar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}') as tqdm_val:
                tqdm_val.set_description(f'Epoch: {epoch}')
                for val_data, labels in tqdm_val:

                    val_data = val_data.to(device)

                    labels = torch.tensor(labels, dtype = torch.long)
                    labels = labels.to(device)

                    prediction = model(val_data).squeeze(1)

                    for idx, i in enumerate(prediction):
                        if torch.argmax(i) == labels[idx]:
                            correct += 1
                        total += 1

                    tqdm_val.set_postfix(accuracy=100.*correct/total)
                    time.sleep(0.1)

            validation_accuracy = round(correct/total, 5)
            print('Validation Accuracy: ', validation_accuracy)
            gc.collect()

        if epoch % save_epoch == 0:
            if not os.path.isdir(save_path):
                os.path.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{epoch}.pt'))

    return model

def genPreds(data, tokenizer, classifier):

    preds = []

    for idx, row in data.iterrows():
        elem = row['Text']
        elem = tokenizer(elem, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = classifier(**elem).logits
        preds.append(logits.argmax().item())

        del elem
        del logits

        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return preds