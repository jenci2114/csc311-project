from utils import *
from torch.autograd import Variable
import csv

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
from meta_process import process_question_meta, get_subject_number



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


class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_students,
        num_subjects,
        question_latent_dim=10,
        subject_latent_dim=5,
        extra_latent_dim=0
        ):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_students, question_latent_dim)
        self.h = nn.Linear(question_latent_dim + subject_latent_dim + extra_latent_dim, num_students)

        self.subject_enc_linear = nn.Linear(num_subjects, subject_latent_dim)

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
        # Forward question entity
        inputs = inputs
        question_raw_latent = sigmoid(self.g(inputs))

        # Beta appending depends on whether beta is given
        if beta is not None:
            question_latent = torch.cat(
                    (question_raw_latent, 
                    torch.tensor([[beta]], dtype=torch.float32).cuda()), 
                    axis=-1) # TODO more modulerized
        else:
            question_latent = question_raw_latent

        # Forward question meta
        if meta is not None:
            meta = meta.cuda()
            subject_latent = sigmoid(self.subject_enc_linear(meta))
            question_full_latent = torch.cat(
                                    (question_latent, subject_latent),
                                    axis=-1
                                    )

        decoded = sigmoid(self.h(question_full_latent))
        return decoded


def train(
    model,
    lr,
    lamb,
    train_data,
    zero_train_data,
    valid_data,
    num_epoch,
    betas=None,
    metas=None
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

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_question = train_data.shape[1]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for question_id in range(num_question):
            inputs = Variable(zero_train_data[:, question_id]).unsqueeze(0).cuda()
            target = inputs.clone()

            optimizer.zero_grad()

            if betas is not None:
                beta = betas[question_id]
            else:
                beta = None

            if metas is not None:
                meta = metas[question_id].unsqueeze(0)
            else:
                meta = None

            output = model(inputs, beta=beta, meta=meta)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[:, question_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # regularizer = 0.5 * lamb * model.get_weight_norm()
            # loss = torch.sum((output - target) ** 2.) + regularizer
            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, betas, metas)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    return model
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data, betas=None, metas=None):
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
        inputs = Variable(train_data[:, q]).unsqueeze(0).cuda()

        if betas is not None:
            beta = betas[q]
        else:
            beta = None

        if metas is not None:
            meta = metas[q].unsqueeze(0)
        else:
            meta = None

        output = model(inputs, beta=beta, meta=meta)

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

    # betas = (betas - 0.5) * 2  # TODO Ways to tune this??
    # betas = None

    # Load raw question metadata
    data = load_train_csv('../data')
    question_count = max(data['question_id']) + 1
    subject_count = get_subject_number('../data/subject_meta.csv')
    raw_question_meta = process_question_meta('../data/question_meta.csv', question_count, subject_count)
    raw_question_meta = torch.FloatTensor(raw_question_meta)

    # Set model hyperparameters.
    # meta_latent_dim = 5
    meta_latent_dim_list = [2, 3, 4, 5, 6, 7]
    k_list = [10, 15, 20, 30]  # 10, 50, 100, 200
    lr_list = [0.01, 0.05, 0.1]  # 0.001, 0.01, 0.1, 1
    epoch_list = [3, 5, 10, 15, 20, 30]  # 3, 5, 10, 15
    lambda_list = [0.001, 0.01, 0.1]
    test_accuracy_list = []
    # Q3, ii, c, tune k, learning rate, and number of epoch
    best_test_accuracy_so_far = 0
    best_parameters = []
    for lamb in lambda_list:
        for meta_latent_dim in meta_latent_dim_list:
            for k in k_list:
                for lr in lr_list:
                    for num_epoch in epoch_list:
                        model = AutoEncoder(
                            num_students=train_matrix.shape[0],
                            num_subjects=subject_count,
                            question_latent_dim=k,
                            subject_latent_dim=meta_latent_dim,
                            extra_latent_dim=1 if betas is not None else 0,
                            ).cuda()
                        train(model, lr, lamb, train_matrix, zero_train_matrix,
                            valid_data, num_epoch, betas=betas, metas=raw_question_meta)
                        test_accuracy = evaluate(model, zero_train_matrix, test_data, betas=betas, metas=raw_question_meta)
                        if test_accuracy > best_test_accuracy_so_far:
                            best_test_accuracy_so_far = test_accuracy
                            best_parameters = [k, lr, num_epoch, meta_latent_dim, lamb]
                        test_accuracy_list.append(test_accuracy)
                        print_string = "k = " + str(k) + " lr = " + str(lr) + " epoch = " + str(num_epoch) + \
                                    " test accuracy = " + str(test_accuracy) + " meta_latent_dim = " + str(meta_latent_dim)
                        print(print_string)
    print("the best parameters I got is: k = " + str(best_parameters[0]) + " learning rate = " + str(best_parameters[1]) + \
          " epoch = " + str(best_parameters[2]) + " meta_latent_dim = " + str(best_parameters[3]) + " best test accuracy is: " + str(best_test_accuracy_so_far +\
            "lambda =" + str(best_parameters[4])))





if __name__ == '__main__':
    # data = load_train_csv('../data')
    # question_count = max(data['question_id']) + 1


    # k = 5
    # lr = 0.1
    # num_epoch = 1500
    # # model = QuestionMetaAE(subject_count, k)
    # # model = train(model, question_meta, lr, num_epoch)
    # # latent_meta = extract_latent(model, question_meta, k)
    # # reconstructed_meta = assess_result(model, question_meta, subject_count)
    torch.manual_seed(2002)
    main()


