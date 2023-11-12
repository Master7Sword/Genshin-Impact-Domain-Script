import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        # 转换为 numpy 数组，并确保数据类型为 float32
        coordinates = self.img_labels.iloc[idx, 1:3].values.astype('float32')
        # 直接将 numpy 数组转换为 torch 张量
        coordinates = torch.from_numpy(coordinates)
        if self.transform:
            image = self.transform(image)
        return image, coordinates

class Residual(nn.Module): 
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 输入通道为3
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Linear(512, 2)  # 最终输出目标点的坐标
)

def train(net, train_loader, num_epochs, learning_rate, device):
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    min_loss = float('inf')
    loss_up_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model_state = net.state_dict()
            loss_up_counter = 0  # reset counter if we have a new min loss
        else:
            loss_up_counter += 1

        if loss_up_counter > 5:
            learning_rate /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            loss_up_counter = 0  # reset counter after reducing learning rate

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Learning Rate: {learning_rate}')

        if learning_rate < 1e-7:  # break if learning rate gets too low
            break

    # Load the best model state before returning
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return net  # return the best model



# 图像预处理
transform = transforms.Compose([
    transforms.Resize((480, 270)),  # 根据模型调整尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':

    # 创建数据集
    train_dataset = CustomImageDataset(
        annotations_file='dataset.txt',
        img_dir='dataset',
        transform=transform
    )

    # 创建数据加载器
    batch_size = 8
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    lr = 0.00001
    num_epochs = 300

    # 调用训练函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_net = train(net, train_loader, num_epochs, lr, device)

    # Save the best model state
    torch.save(best_net.state_dict(), 'best_model_weights.pth')
