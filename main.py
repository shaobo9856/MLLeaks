import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ShadowModel, AttackModel
import torch.optim as optim
import torch.nn.functional as F

    
def load_data(dataset_name, batch_size):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_target_model(train_loader, model, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def get_top3_probabilities(model, data_loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            all_probs.append(probabilities)

    return torch.cat(all_probs, dim=0)  # 合并所有批次的概率

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()
    print(f'Dataset: {args.dataset}')
    
    # Step 0: Split the data into DShadow_train and DShadow_out
    if args.dataset == 'CIFAR10':
        num_classes = 10
        train_loader, out_loader = load_data(args.dataset, args.batch_size)
    elif args.dataset == 'MNIST':
        num_classes = 10
        train_loader, out_loader = load_data(args.dataset, args.batch_size)
    else:
        raise ValueError('Invalid dataset name')

    # Step 1: Train the Shadow model on DShadow_train
    shadowmodel = ShadowModel(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(shadowmodel.parameters(), lr=0.001)
    train_target_model(train_loader, shadowmodel, criterion, optimizer, args.num_epochs)

    # Step 2: Construct Attact datasets
    # Get the top 3 probabilities from the shadow model
    DShadow_train_y = get_top3_probabilities(shadowmodel, train_loader)
    DShadow_out_y = get_top3_probabilities(shadowmodel, out_loader)

    # Construct datasets for attack model training
    DShadow_train_y = DShadow_train_y[:, :3]  # Top 3 probabilities for DShadow_train
    DShadow_out_y = DShadow_out_y[:, :3]  # Top 3 probabilities for DShadow_out

    # Create labels and Combine the two datasets
    DShadow_train_labels = torch.ones(DShadow_train_y.size(0))  # Label 1 for DShadow_train
    DShadow_out_labels = torch.zeros(DShadow_out_y.size(0))  # Label 0 for DShadow_out
    combined_y = torch.cat((DShadow_train_y, DShadow_out_y), dim=0)
    combined_labels = torch.cat((DShadow_train_labels, DShadow_out_labels), dim=0)

    # Prepare data for attack model training
    attack_dataset = torch.utils.data.TensorDataset(combined_y, combined_labels)
    attack_loader = DataLoader(attack_dataset, batch_size=args.batch_size, shuffle=True)

    # Step 3: Attack Model Training on the combined dataset
    attack_model = AttackModel(input_size=combined_y.size(1))  # 128/ Assuming input_size is the number of features / attack_model = AttackModel(input_size=128)
    attack_criterion = torch.nn.CrossEntropyLoss()
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    train_target_model(attack_loader, attack_model, attack_criterion, attack_optimizer, args.num_epochs)

    # Step 4: Evaluate the attack model on the test set
    print("done")

if __name__ == '__main__':
    main()
