

# Reproduce Code for Paper "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models" 
Only for Adversary 1

![alt text](https://cdn-fusion.imgcdn.store/i/2024/2106384aa5ba04ed.jpeg)

## Installation

pip install -r requirements.txt

## Run experiments

#### For MNIST dataset: 

python3 main.py --dataset MNIST --num_epochs 5 --batch_size 64

#### For CIFAR10 dataset: 

python3 main.py --dataset CIFAR10 --num_epochs 5 --batch_size 100



## Process
Step 0. Split DShadow dataset(MNIST, CIFAR10) into DShadow_train and DShadow_out

Step 1. Shadow Model Training
        
        Input: DShadow dataset(DShadow_train_x)

        Output: DShadow dataset(DShadow_train_y)

Step 2. Shadow Model Testing(Obtained DShadow_train_y,DShadow_out_y ), build Attack Dataset

Step 3. Attack Model Training

        Input:  DShadow_train_y, DShadow_out_y   #(top 3 probabilities)

        Output: 1(DShadow_train_y), 0(DShadow_out_y)

Step 4. Membership Inference(Testing, DTarget_y, DShadow_train_y )

        Model: Attack Model

        Input: DTarget_y #(top 3 probabilities)

        Output: 1(DTarget_y), 0(out of DTarget)




​
 





