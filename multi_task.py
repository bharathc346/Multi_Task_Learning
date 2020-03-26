import json
import numpy as np
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from generate_data import generate_datasets
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# parameters
LEARNING_RATE = 0.01
REG_PARAM = 1.0
NUM_DOCS = 250
NUM_EPOCHS = 100
BATCH_SIZE = 16
VOCAB_SIZE = 1000
RANDOM_SEED = 2
TEST_SPLIT = 0.3

class SyntheticDataset(data.Dataset):

    def __init__(self, raw_dataset, vocab_size=VOCAB_SIZE):
        self.set = []
        self.vocab_size = vocab_size
        self.labels = raw_dataset['labels']
        self.features = raw_dataset['features']
        self.gen_data()

    def get_feature_matrix(self, dataset):
        feat_matrix = []
        for doc in dataset:
            x = [0 for i in range(1000)]
            for token in doc:
                x[token] = doc[token]
            feat_matrix.append(x)
        return torch.FloatTensor(feat_matrix)

    def gen_data(self):
        for doc_feats, doc_label in zip(self.features, self.labels):
            dense_feat = [0 for i in range(self.vocab_size)]
            for token in doc_feats:
                dense_feat[token] = doc_feats[token]
            doc_label = 0 if doc_label == -1 else 1
            self.set.append(
                (torch.FloatTensor(dense_feat), torch.LongTensor([doc_label]))
            )

    def __getitem__(self, idx):
        return self.set[idx]

    def __len__(self):
        return len(self.set)

class LogisticRegression(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(vocab_size, 2)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)

''' Reset model parameters to random initial condition '''
def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

''' Generate bar plot with mean and standard devaitions '''
def gen_bar_plot(res, title,task_i):
    means = [statistics.mean(res[k]) for k in res]
    positions = [0, 1, 2, 3, 4]
    std = [statistics.stdev(res[k]) for k in res]
    plt.subplot(3,1, task_i)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Prop_Prior_Overlaps')
    plt.xticks(positions, ('0.0', '0.25', '0.5', '0.75', '1.0'))
    plt.bar(positions, means, width=0.35, yerr=std, alpha=0.5, ecolor='black', capsize=10)

''' Pretrains on the three datasets excluding dataset_i within a prior
    and returns parameter vector theta'''
def pretrain(model, optimizer, total_data, prop_prior_overlap, dataset_i):
    # generate raw dataset of remaining 3 datasets
    raw_dataset = {
        'labels': [],
        'features': []
    }
    for i in range(4):
        if dataset_i == i:
            continue
        curr_dataset = total_data[prop_prior_overlap][dataset_i]

        raw_dataset['labels'] += curr_dataset['labels']
        raw_dataset['features'] += curr_dataset['features']

    train_set = SyntheticDataset(raw_dataset)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    theta = train(model, optimizer, train_loader)
    return theta

''' Train loop of model, takes keyword argument theta_0 if
    special regularization is needed'''
def train(model, optimizer, train_loader, theta_0=None):
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = batch
            y = y.view(-1) # get rid of batch dimension

            log_probs = model(x)
            train_loss = loss_function(log_probs, y)
            if theta_0 is not None:
                train_loss = train_loss + REG_PARAM * (torch.norm(model.linear.weight - theta_0) ** 2)

            train_loss.backward()
            optimizer.step()
    theta = model.linear.weight
    return theta

''' Test loop of model, returns accuracy '''
def test(model, test_loader):
    correct = 0
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            x, y = batch
            y = y.view(-1)
            log_probs = model(x)
            _, predicted = torch.max(log_probs.data, 1)
            correct += (predicted == y).float().sum()
    return 100 * correct / (TEST_SPLIT * NUM_DOCS)

''' Splits a dataset_i into train and test and returns their data loaders '''
def split_train_test(total_data, prop_prior_overlap, dataset_i):
    dataset = SyntheticDataset(total_data[prop_prior_overlap][dataset_i])
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(TEST_SPLIT * dataset_size))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    return train_loader, test_loader


if __name__ == '__main__':
    # get total data (5 priors each with 4 datsets) from data generation script
    total_data = generate_datasets(N_features=1000, N_instances=250, N_datasets= 4, class_overlap= 0.25)

    results = {1: {}, 2: {}, 3: {}}

    for task_i in range(1,4):
        # init model, loss, and optimizer
        model = LogisticRegression()
        loss_function = nn.NLLLoss()
        if task_i == 3:
            # specificy custom regularization in train function rather than in optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=REG_PARAM)

        for prop_prior_overlap in np.arange(0, 1.25, 0.25):
            prop_prior_accuracies = []
            for dataset_i in range(4):
                # reset model weights everytime in vanilla logistic regression
                if task_i == 1:
                    model.apply(weight_reset)
                else:
                    theta = pretrain(model, optimizer, total_data, prop_prior_overlap, dataset_i)

                train_loader, test_loader = split_train_test(total_data, prop_prior_overlap, dataset_i)
                if task_i == 3:
                    train(model, optimizer, train_loader, theta)
                else:
                    train(model, optimizer, train_loader)
                dataset_accuracy = test(model, test_loader)
                prop_prior_accuracies.append(dataset_accuracy.item())

            results[task_i][prop_prior_overlap] = prop_prior_accuracies


    with open("dataset_accuracies.json", "w") as f:
        json.dump(results, f, indent=4)
    # print(json.dumps(results, indent=4))
    titles = {1:'Task 1: Implement Vanilla Logistic Regression', 2:'Task 2: Estimate initial parameter based on other datasets', 3:'Task 3: Shared parameter objective'}

    # Graph results
    fig, _ = plt.subplots(3, 1)
    for task_i in results:
        gen_bar_plot(results[task_i], titles[task_i], int(task_i))

    fig.subplots_adjust(hspace=1)
    plt.show()
