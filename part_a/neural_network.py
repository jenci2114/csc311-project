from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch

from torch import sigmoid

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
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encoded = sigmoid(self.g(inputs))
        decoded = sigmoid(self.h(encoded))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
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
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # the three lists below are used to plot
    epoch_list = []
    training_loss_list = []
    validation_accuracy_list = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            model.train()
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            regularizer = 0.5 * lamb * model.get_weight_norm()
            loss = torch.sum((output - target) ** 2.) + regularizer
            # loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            # for part 3 ii d
            # It compute validation objective. But according to Piazza we should plot valid accuracy
            # So I comment the code below
            # with torch.no_grad():
            #     model.eval()
            #     for i, u in enumerate(valid_data["user_id"]):
            #         valid_inputs = Variable(zero_train_data[u]).unsqueeze(0)
            #         valid_output = model(valid_inputs)
            #         valid_guess = valid_output[0][valid_data["question_id"][i]].item()
            #         valid_loss += (valid_guess - valid_data["is_correct"][i]) ** 2.

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        epoch_list.append(epoch)
        training_loss_list.append(train_loss)
        validation_accuracy_list.append(valid_acc)

    # Plot for Q3d
    # plt.plot(epoch_list, training_loss_list)
    # plt.xlabel("epoch number")
    # plt.ylabel("training loss")
    # plt.title("epoch vs training loss")
    # plt.show()
    #
    # plt.plot(epoch_list, validation_accuracy_list)
    # plt.xlabel("epoch number")
    # plt.ylabel("validation accuracy")
    # plt.title("epoch vs validation accuracy")
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
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

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k_list = [10, 50, 100, 200, 500]
    lr_list = [0.001, 0.01, 0.1, 1]
    epoch_list = [3, 5, 10, 15]
    valid_accuracy_list = []

    # Q3, ii, c, tune k, learning rate, and number of epoch
    # Don't forget to remove the regularizer before running Q3 ii c
    lamb = 0.001
    best_valid_accuracy_so_far = 0
    best_parameters = []
    for k in k_list:
        for lr in lr_list:
            for num_epoch in epoch_list:
                model = AutoEncoder(train_matrix.shape[1], k)
                train(model, lr, lamb, train_matrix, zero_train_matrix,
                      valid_data, num_epoch)
                valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
                if valid_accuracy > best_valid_accuracy_so_far:
                    best_valid_accuracy_so_far = valid_accuracy
                    best_parameters = [k, lr, num_epoch]
                valid_accuracy_list.append(valid_accuracy)
                print_string = "k = " + str(k) + " lr = " + str(lr) + " epoch = " + str(num_epoch) + \
                               " valid accuracy = " + str(valid_accuracy)
                print(print_string)
    print("the best parameters I got is: k = " + str(best_parameters[0]) + " learning rate = " + str(best_parameters[1]) + \
          " epoch = " + str(best_parameters[2]) + " valid accuracy = ", best_valid_accuracy_so_far)

    valid_accuracy_for_k = []
    k_list = [10, 50, 100, 200, 500]
    lr = 0.1
    num_epoch = 15
    lamb = 0.001
    for k in k_list:
        model = AutoEncoder(train_matrix.shape[1], k)
        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch)
        valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
        valid_accuracy_for_k.append(valid_accuracy)
    plt.plot(k_list, valid_accuracy_for_k)
    plt.xlabel("k")
    plt.ylabel("validation accuracy")
    plt.title("k vs validation accuracy")
    plt.show()

    print("lambda list is ", k_list, ", accuracy list is ", valid_accuracy_for_k)

    # Q3, ii, d
    # Don't forget to remove the regularizer before running Q3 ii d
    lamb = 0.001
    k = 10
    lr = 0.1
    num_epoch = 15

    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print_string = "k = " + str(k) + " lr = " + str(lr) + " epoch = " + str(num_epoch) + \
                   " test accuracy = " + str(test_accuracy)
    print(print_string)


    # Q3, ii, e
    k = 10
    lr = 0.1
    num_epoch = 15
    lambda_list = [0.001, 0.01, 0.1, 1]
    accuracy_list = []
    best_test_accuracy_so_far = 0
    best_parameters = 0
    for lamb in lambda_list:
        model = AutoEncoder(train_matrix.shape[1], k)
        train(model, lr, lamb, train_matrix, zero_train_matrix,
              valid_data, num_epoch)
        valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
        if valid_accuracy > best_test_accuracy_so_far:
            best_test_accuracy_so_far = valid_accuracy
            best_parameters = lamb
        accuracy_list.append(valid_accuracy)
        print_string = "lambda = " + str(lamb) + " valid accuracy = " + str(valid_accuracy)
        print(print_string)
    print("best lambda is "  + str(best_parameters) + " best accuracy is "  + str(best_test_accuracy_so_far))

    lamb = best_parameters
    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    final_valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
    final_test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print("With our chosen lambda, the validation accuracy is ", str(final_valid_accuracy), ", the test accuracy is ",
          str(final_test_accuracy))

    plt.plot(lambda_list, accuracy_list)
    plt.xlabel("lambda")
    plt.ylabel("validation accuracy")
    plt.title("lambda vs validation accuracy")
    plt.show()

    print("lambda list is ", lambda_list, ", accuracy list is ", accuracy_list)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    torch.manual_seed(208)
    main()
