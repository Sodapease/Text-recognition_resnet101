import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

def create_data_loaders(data_root, batch_size=64, test_size=0.2, num_workers=0):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 创建数据集对象
    dataset = ImageFolder(root=data_root, transform=transform)

    # 划分训练集和测试集
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)

    # 创建训练集和测试集的数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset, train_dataset, test_dataset, train_dataloader, test_dataloader

def load_resnet101_model(num_classes, device):
    model = models.resnet101(pretrained=True)
    # 替换模型的全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model


def define_loss_optimizer(model, learning_rate):
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器，并关联模型参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device):
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to train mode
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc="Epoch {}/{}".format(epoch + 1, num_epochs))

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': running_loss / (batch_idx + 1)})

        # Record the loss
        average_loss = running_loss / len(train_dataloader)
        losses.append(average_loss)

        # Calculate accuracy at the end of each epoch
        model.eval()  # Set the model to evaluation mode
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                num_total += labels.size(0)
                num_correct += (predicted == labels).sum().item()

        accuracy = num_correct / num_total
        accuracies.append(accuracy)

        print('[Epoch {}/{}], Average Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, average_loss,
                                                                             accuracy))

    return losses, accuracies

def plot_accuracy(accuracies, num_epochs):
    plt.plot(range(1, num_epochs+1), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.grid()
    plt.show()

def plot_loss(losses, num_epochs):
    plt.plot(range(1, num_epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.grid()
    plt.show()


def predict_samples(model, test_dataloader, device, class_names):
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # 输出部分模型预测结果
    for i in range(len(predictions)):
        predicted_class = class_names[predictions[i]]
        probability = probabilities[i]
        max_probability = np.max(probability)
        print(f"image {i + 1}: pred_cls: {predicted_class}, probability: {max_probability}")

    return probabilities


def compute_mAP(ground_truths, probabilities, num_classes):
    average_precisions = []

    for class_index in range(num_classes):
        class_ground_truths = (ground_truths == class_index).astype(int)
        class_probabilities = probabilities[:, class_index]

        sorted_indices = np.argsort(class_probabilities)[::-1]
        sorted_ground_truths = class_ground_truths[sorted_indices]

        true_positives = np.cumsum(sorted_ground_truths)
        false_positives = np.cumsum(1 - sorted_ground_truths)
        denominator = true_positives + false_positives
        denominator[denominator == 0] = 1.0  # 将分母为零的位置设置为 1.0，以避免除法错误
        precision = true_positives / denominator

        denominator = np.sum(class_ground_truths)
        if denominator == 0:
            recall = np.zeros_like(true_positives)
        else:
            recall = true_positives / denominator
        recall = np.insert(recall, 0, 0.0)

        if np.sum(class_ground_truths) == 0:
            average_precision = 0.0
        else:
            average_precision = np.sum((recall[1:] - recall[:-1]) * precision)
        average_precisions.append(average_precision)

    mAP = np.mean(average_precisions)
    return mAP


if __name__ == "__main__":
    data_root = './dataset-fonts'
    batch_size = 64
    dataset, train_dataset, test_dataset, train_dataloader, test_dataloader = create_data_loaders(data_root, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)
    model = load_resnet101_model(num_classes, device)

    learning_rate = 0.001
    criterion, optimizer = define_loss_optimizer(model, learning_rate)

    num_epochs = 10
    losses, accuracies = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device)

    plot_accuracy(accuracies, num_epochs)
    plot_loss(losses, num_epochs)

    folder_list = glob.glob('dataset-fonts/*')

    num_classes = len(folder_list)

    # 调用函数计算mAP
    ground_truths = [label for _, label in test_dataset]
    ground_truths = np.array(ground_truths)
    class_names = dataset.classes
    probabilities = predict_samples(model, test_dataloader, device, class_names)
    probabilities = np.array(probabilities)
    mAP = compute_mAP(ground_truths, probabilities, num_classes)
    print("mAP:", mAP)