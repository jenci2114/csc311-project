from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch

import math
from torch import sigmoid
import item_response

from torch.utils.data import DataLoader, Dataset, random_split


def read_encoded_question_metadata(filepath, question_num, k):
    """Read the encoded question metadata from <filepath>."""
    res = np.zeros((question_num, k))
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            q_id = int(row[0])
            res[q_id] = np.array(row[1:]).astype(float)
    return res

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AEDataset(Dataset):
    def __init__(self, zero_train_matrix, beta_vector, metadata) -> None:
        """
        args:
            - zero_train_matrix: table with nan replaced by 0.
        """
        super().__init__()
        self.zero_train_matrix = zero_train_matrix
        self.beta_vector = beta_vector
        self.metadata = metadata
        
        
    def __len__(self):
        return self.zero_train_matrix.shape[1] 
        
    def __getitem__(self, idx):
        """
        args:
            - idx: question idx
        """
        return {'question_id': idx,
                'question_vector': self.zero_train_matrix[:, idx],
                'beta': torch.tensor([self.beta_vector[idx]], dtype=torch.float32) 
                        if self.beta_vector is not None else torch.nan,
                'meta_latent': self.metadata[idx] 
                        if self.metadata is not None else torch.nan
                }
    
    
class AutoEncoder(nn.Module):
    def __init__(self, num_students, k=100, extra_latent_dim=0):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_students, k)
        self.h = nn.Linear(k + extra_latent_dim, num_students)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def get_raw_latent(self, inputs):
        return sigmoid(self.g(inputs))

    def forward(self, inputs, beta=None, meta=None):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        question_raw_latent = sigmoid(self.g(inputs))
        if beta is not None:
            # question_latent = torch.cat(
            #         (question_raw_latent, torch.tensor([[beta]], dtype=torch.float32)), axis=-1) # TODO more modulerized
            # beta = beta.reshape(-1, 1)
            question_latent = torch.cat((question_raw_latent, beta), axis=-1)
        else:
            question_latent = question_raw_latent
        
        if meta is not None:
            question_latent = torch.cat((question_latent, meta), axis=-1)
        
        decoded = sigmoid(self.h(question_latent))
        return decoded


def train(
    model,
    lr,
    batch_size,
    lamb,
    train_data,
    zero_train_data,
    valid_data,
    num_epoch,
    betas,
    metadata
    ):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()
    
    # Build dataset object
    dataset = AEDataset(
        zero_train_matrix=zero_train_data, 
        beta_vector=betas,
        metadata=metadata
        )
    
    # Define dataloader 
    dataloarder = DataLoader(
                            dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=True
                            )

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # num_question = train_data.shape[1]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for datapoints in dataloarder:
            # inputs = Variable(zero_train_data[:, question_id]).unsqueeze(0)
            
            question_id_batch = datapoints['question_id']
            question_vectors_batch = datapoints['question_vector']
            beta_batch = datapoints['beta']
            meta_batch = datapoints['meta_latent']
            
            inputs = Variable(question_vectors_batch)   # TODO: need Variable()?
            targets = inputs.clone()

            optimizer.zero_grad()
            # if betas is not None:
            #     beta = betas[question_id]
            #     output = model(inputs, beta)
            # else:
            #     output = model(inputs)
            
            outputs = model(
                        inputs, 
                        beta=beta_batch if not torch.isnan(beta_batch.flatten()[0]).item() else None,
                        meta=meta_batch if not torch.isnan(meta_batch.flatten()[0]).item() else None,
                        )

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[:, question_id_batch].numpy()).T   # TODO: torch isnan enough?
            nan_mask = torch.tensor(nan_mask)


            targets[nan_mask] = outputs[nan_mask]

            regularizer = 0.5 * lamb * model.get_weight_norm()
            loss = torch.sum((outputs - targets) ** 2.) + regularizer
            loss = torch.sum((outputs - targets) ** 2.)              # TODO: sum of all, mean or not (for adam not matter that much)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, betas, metadata)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    return model
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data, betas, metadata):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, q in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[:, q]).unsqueeze(0)
        # if betas is not None:
        #     betas = torch.tensor([betas[q]], dtype=torch.float32)
        # else:
        #     betas = None
        
        # output = model(inputs, betas)
        # if metadata is not None:
        #     output = model(inputs, torch.tensor(metadata[q], dtype=torch.float32))
        
        
        # output = model(inputs, betas=betas, metadata=metadata)
        output = model(
            inputs=inputs,
            beta=torch.tensor([[betas[q]]], dtype=torch.float32) if betas is not None else None,
            meta=metadata[q].unsqueeze(0) if metadata is not None else None
        )


        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def get_latent_mat(model, zero_train_data, entity='question'):
    if entity == 'question':
        batched_input = torch.t(zero_train_data)
        batched_latent = model.get_raw_latent(batched_input)
        latent_mat = torch.t(batched_latent)
        breakpoint()
        return latent_mat.detach().numpy()




def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Pre-train IRT model
    _, betas, _, _, _ = \
        item_response.irt(
        data=train_matrix.detach().numpy(),
        val_data=valid_data,
        lr=0.01,
        iterations=25,
    )

    question_num = train_matrix.shape[1]
    k_meta = 5
    metadata = read_encoded_question_metadata('../data/question_meta_encoded.csv', question_num, k_meta)
    metadata = torch.tensor(metadata, dtype=torch.float32)

    
    # betas = None
    batch_sizes = [5, 10, 30]

    # Set model hyperparameters.
    k_list = [3, 5, 8, 10]  # 10, 50, 100, 200
    lr_list = [1e-2, 1e-3, 1e-4]  # 0.001, 0.01, 0.1, 1
    epoch_list = [3, 5, 10, 15, 20, 25]  # 3, 5, 10, 15
    test_accuracy_list = []
    # Q3, ii, c, tune k, learning rate, and number of epoch
    lamb = 0
    best_test_accuracy_so_far = 0
    best_hp = {}
    for batch_size in batch_sizes:
        for k in k_list:
            for lr in lr_list:
                for num_epoch in epoch_list:
                    extra_latent_dim = 1 if betas is not None else 0
                    extra_latent_dim += 5 if metadata is not None else 0
                    model = AutoEncoder(
                        train_matrix.shape[0], 
                        k, 
                        extra_latent_dim=extra_latent_dim)
                    
                    train(
                        model, 
                        lr, 
                        batch_size,
                        lamb, 
                        train_matrix, 
                        zero_train_matrix,
                        valid_data, 
                        num_epoch, 
                        betas,
                        metadata
                        )
                    test_accuracy = evaluate(model, zero_train_matrix, test_data, betas, metadata)
                    if test_accuracy > best_test_accuracy_so_far:
                        best_test_accuracy_so_far = test_accuracy
                        best_hp = {
                                    'k': k, 
                                    'bs': batch_size,
                                    'lr': lr,
                                   'epoch': num_epoch
                                   } 
                    test_accuracy_list.append(test_accuracy)
                    # print_string = \
                    #             "k = " + str(k) + \
                    #             ",   batch_size = {batch_size}",
                    #             ",   lr = " + str(lr) + \
                    #             ",   epoch = " + str(num_epoch) + \
                    #             ",   test accuracy = " + str(test_accuracy)
                    # print(print_string)
                    print(f"k = {k},  bs = {batch_size},  lr = {lr},  epoch = {num_epoch},  test_accuracy = {test_accuracy}")
                    # dict = {'k': k, 'bs': batch_size, 'lr': lr, 'num_epoch': num_epoch, 'test_accuracy': test_accuracy}
                    # print(dict)
    print(f"Best parameters: k = {best_hp['k']},  bs = {best_hp['bs']},  lr = {best_hp['lr']},  epoch = {best_hp['num_epoch']},  best test accuracy is = {best_test_accuracy_so_far}" )
    # plt.plot(k_list, test_accuracy_list)
    # plt.xlabel("k value")
    # plt.ylabel("test accuracy")
    # plt.title(title)
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    torch.manual_seed(2002)
    main()
