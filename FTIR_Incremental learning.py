import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import PIL
import torch
import torchvision.transforms as transforms
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Softmax, Conv1d
warnings.filterwarnings('ignore')


class Net(torch.nn.Module):
    def __init__(self, classes):
        super(Net, self).__init__()
        #conv_layer_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        conv_layer_1 = Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        # conv_layer_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # conv_layer_3 = Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        # conv_layer_4 = Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        conv_layer_2 = Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        conv_layer_3 = Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        conv_layer_4 = Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        pooling_layer = MaxPool2d(kernel_size=2, stride=2)
        activation = ReLU()
        final_activation = Softmax()

        conv_set = [conv_layer_1, activation, pooling_layer,
                    conv_layer_2, activation, pooling_layer,
                    conv_layer_3, activation, pooling_layer,
                    conv_layer_4, activation, pooling_layer]

        self.conv_layers = torch.nn.Sequential(*conv_set)

        linear_layer_1 = Linear(10368, 512)
        linear_layer_2 = Linear(512, 256)

        linear_set = [linear_layer_1, activation,
                      linear_layer_2, activation]

        self.linear_layers = torch.nn.Sequential(*linear_set)
        self.final_linear_layer = torch.nn.Sequential(Linear(256, classes), final_activation)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        x = self.final_linear_layer(x)
        return x


increment = False
base_classes_trained = 4
step_increment_classes = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = base_classes_trained
model = Net(classes).to(device)
print(model)

if increment:
    classes = base_classes_trained + step_increment_classes
    model = torch.load("/kaggle/input/network/model")
    model.conv_layers.requires_grad_(False)
    model.linear_layers.requires_grad_(False)

    new_linear1 = Linear(256, 64)
    activation = ReLU()

    final_linear = Linear(64, classes)
    final_activation = Softmax()

    modified_layers = [new_linear1, activation, final_linear, final_activation]
    model.final_linear_layer = torch.nn.Sequential(*modified_layers)
    model = model.to(device)
    print("Modified Network", model)
df = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
df["filename"] = df['id'] + ".jpg"

mini_df = df.groupby('breed').apply(lambda s: s.sample(60))
samples = 60 - 1
mini_batch = mini_df.iloc[:(samples * classes)]

if increment:
    image_name = mini_df.iloc[samples * classes].filename
    image_breed = mini_df.iloc[samples * classes].breed
    print(f"train incremental learning for breed: '{image_breed}'")
    image = cv2.imread("/kaggle/input/dog-breed-identification/train/" + image_name)
    plt.imshow(image)
    plt.title(f"{image_breed}")
    plt.show()

width, height, channels = image.shape

# new_df =mini_df.iloc[180:239]

print(len(mini_df), len(mini_batch))
mini_batch.describe()

print("Train Model for breeds:")
for i in range(classes):
    image_name = mini_batch.iloc[i * 60].filename
    image_breed = mini_batch.iloc[i * 60].breed
    image = cv2.imread("/kaggle/input/dog-breed-identification/train/" + image_name)
    plt.imshow(image)
    plt.title(f"{image_breed}")
    plt.show()
train_df, valid_df = train_test_split(mini_batch, test_size=0.15, random_state=10)

transformation = transforms.Compose([
    transforms.Resize(size=(300, 300)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DataCreation(torch.utils.data.Dataset):
    def __init__(self, data, encoding, transformation):
        self.data = data
        self.transformation = transformation
        self.encoding = encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        loc = "/kaggle/input/dog-breed-identification/train/" + f"{self.data.iloc[index].filename}"
        image = PIL.Image.open(loc)
        image = self.transformation(image)
        image_name = self.data.iloc[index].breed
        label = self.encoding.transform([[image_name]])
        return (image, label)


batch_size = 4
epochs = 100

le = LabelEncoder()
label = mini_df.breed.unique()[:classes].reshape(-1, 1)
label_encoding = le.fit_transform(label)

train_data = DataCreation(train_df, le, transformation)
val_data = DataCreation(valid_df, le, transformation)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()


def training(epochs, trainloader, model, criterion, optimizer, device):
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    for i in range(epochs):
        loss_1 = 0
        correct_pred = 0
        total_samples = 0
        val_loss_1 = 0
        val_correct_pred = 0
        val_total_samples = 0
        for data in trainloader:
            img, label = data
            img = img.to(device)
            label = torch.tensor(label, dtype=torch.long).to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label.view(-1))
            loss_1 += loss.item()
            loss.backward()
            optimizer.step()

            _, pred_label = pred.max(dim=1)
            pred_label = pred_label.type(dtype=torch.float)
            label = (torch.tensor(label, dtype=torch.float).view(-1)).to(device)
            total_samples += len(label)
            correct_pred += (pred_label == label).sum()

        with torch.no_grad():
            for data in valloader:
                img, label = data
                img = img.to(device)
                label = torch.tensor(label, dtype=torch.long).to(device)
                pred = model(img)
                loss = criterion(pred, label.view(-1))
                _, pred_label = pred.max(dim=1)
                pred_label = pred_label.type(dtype=torch.float)
                label = (torch.tensor(label, dtype=torch.float).view(-1)).to(device)
                val_loss_1 += loss.item()
                val_total_samples += len(label)
                val_correct_pred += (pred_label == label).sum()

        train_loss = loss_1 / len(trainloader)
        train_accuracy = 100 * (int(correct_pred) / total_samples)
        val_loss = val_loss_1 / len(valloader)
        val_accuracy = 100 * (int(val_correct_pred) / val_total_samples)

        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print("Epoch: ", i)
        print("correct_prediction: ", int(correct_pred))
        print("total_samples", total_samples)
        print("Loss :", train_loss)
        print("correct_prediction_percentage is:", train_accuracy)
        print("\n")
        print("correct_prediction val: ", int(val_correct_pred))
        print("val_total_samples", val_total_samples)
        print("val_Loss :", val_loss)
        print("val correct_prediction_percentage is:", val_accuracy)
        print("\n")
        print("\n")

    # save the network
    torch.save(model, "model")
    return train_acc_list, val_acc_list, train_loss_list, val_loss_list


train_accuracy, val_accuracy, train_loss, val_loss = training(epochs, trainloader, model, criterion, optimizer, device)

epoch_range = torch.linspace(0, epochs-1)
plt.plot(epoch_range, train_accuracy)
plt.plot(epoch_range, val_accuracy)
plt.savefig('accuracy_4_classes.png')
plt.show()


plt.plot(epoch_range, train_loss)
plt.plot(epoch_range, val_loss)
plt.savefig('Loss_4_classes.png')
plt.show()