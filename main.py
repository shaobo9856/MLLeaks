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

    # For train_dataset
    train_size = int(0.8 * len(train_dataset))  # 80% for training
    out_size = len(train_dataset) - train_size  # 20% for out-of-training
    train_in, train_out = torch.utils.data.random_split(train_dataset, [train_size, out_size])

    # For test_dataset
    test_size = int(0.8 * len(test_dataset))  # 80% for testing
    test_out_size = len(test_dataset) - test_size  # 20% for out-of-testing
    test_in, test_out = torch.utils.data.random_split(test_dataset, [test_size, test_out_size])

    train_in_loader = DataLoader(train_in, batch_size=batch_size, shuffle=True, drop_last=True)
    train_out_loader = DataLoader(train_out, batch_size=batch_size, shuffle=False, drop_last=True)
    test_in_loader = DataLoader(test_in, batch_size=batch_size, shuffle=True, drop_last=True)
    test_out_loader = DataLoader(test_out, batch_size=batch_size, shuffle=False, drop_last=True)
        
    return train_in_loader, train_out_loader,test_in_loader,test_out_loader

def train_target_model(train_loader, model, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        i = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            print(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}, Outputs shape: {outputs.shape}')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}], data [{i}], Loss: {loss.item():.4f}')
            i += 1
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def get_probabilities(model, data_loader):
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
    
    # Step 0: Split the data into train_in_loader, train_out_loader,test_in_loader,test_out_loader
    if args.dataset == 'CIFAR10':
        num_classes = 10
        is_cifar = True
        train_in_loader, train_out_loader,test_in_loader,test_out_loader = load_data(args.dataset, args.batch_size)
    elif args.dataset == 'MNIST':
        num_classes = 10
        is_cifar = False
        train_in_loader, train_out_loader,test_in_loader,test_out_loader = load_data(args.dataset, args.batch_size)
    else:
        raise ValueError('Invalid dataset name')

    # Step 1: Train the Shadow model on train_in dataset
    shadowmodel = ShadowModel(num_classes=num_classes, is_cifar=is_cifar)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(shadowmodel.parameters(), lr=0.001)
    train_target_model(train_in_loader, shadowmodel, criterion, optimizer, args.num_epochs)

    # Step 2: Construct Attact datasets
    # Get the top 3 probabilities from the shadow model
    print("11111111111")
    train_in_y = get_probabilities(shadowmodel, train_in_loader)
    train_out_y = get_probabilities(shadowmodel, train_out_loader)
    train_in_y = train_in_y[:, :3]  # get top 3 probabilities for DShadow_train
    train_out_y = train_out_y[:, :3]  # get top 3 probabilities for DShadow_out
    print("22222222222")

    # Create labels and Combine the two datasets
    train_in_labels = torch.ones(train_in_y.size(0), dtype=torch.long)  # Label 1 for train_in
    train_out_labels = torch.zeros(train_out_y.size(0), dtype=torch.long)  # Label 0 for train_out
    combined_y = torch.cat((train_in_y, train_out_y), dim=0)
    combined_labels = torch.cat((train_in_labels, train_out_labels), dim=0)
    print("333333333")

    # Prepare attack dataset for attack model training
    attack_dataset = torch.utils.data.TensorDataset(combined_y, combined_labels.to(torch.long))
    attack_loader = DataLoader(attack_dataset, batch_size=args.batch_size, shuffle=True)
    print("44444444")

    # Step 3: Attack Model Training on the combined attack dataset
    attack_model = AttackModel(input_size=combined_y.size(1))  # 128 / Assuming input_size is the number of features
    attack_criterion = torch.nn.CrossEntropyLoss()
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    print("555555555")
    train_target_model(attack_loader, attack_model, attack_criterion, attack_optimizer, args.num_epochs)

    # Step 4: Evaluate the attack model on the test set
    target_model =  ShadowModel(num_classes=num_classes, is_cifar=is_cifar)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(shadowmodel.parameters(), lr=0.001)
    train_target_model(test_in_loader, target_model, criterion, optimizer, args.num_epochs)

    # get prob of test_in  
    test_in_probs = get_probabilities(target_model, test_in_loader)
    test_in_probs = test_in_probs[:, :3]  # Get top 3 probabilities

    # get prob of test_out
    test_out_probs = get_probabilities(target_model, test_out_loader)
    test_out_probs = test_out_probs[:, :3]  # Get top 3 probabilities

    with torch.no_grad():
        attack_outputs_in = attack_model(test_in_probs)
        attack_outputs_out = attack_model(test_out_probs)

        # Get the predicted labels for test_in and test_out
        _, predicted_labels_in = torch.max(attack_outputs_in, dim=1)
        _, predicted_labels_out = torch.max(attack_outputs_out, dim=1)

    # true labels from test_in (all should be 1) and test_out (all should be 0)
    true_labels_in = torch.ones(predicted_labels_in.size(0), dtype=torch.long) 
    true_labels_out = torch.zeros(predicted_labels_out.size(0), dtype=torch.long)

    # Combine results for accuracy calculation
    predicted_labels = torch.cat((predicted_labels_in, predicted_labels_out))
    true_labels = torch.cat((true_labels_in, true_labels_out))

    # Calculate accuracy
    accuracy = (predicted_labels == true_labels).float().mean().item() * 100
    print(f'Accuracy of the attack model on the test set: {accuracy:.2f}%')


if __name__ == '__main__':
    main()