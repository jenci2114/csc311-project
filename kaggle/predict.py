"""
This module generates files that can be used to submit
to the CSC311 Kaggle competition.
"""
from part_a.item_response import irt
from part_b.ae import AutoEncoder, train, evaluate
from utils import load_train_sparse, load_valid_csv, load_public_test_csv, \
    load_private_test_csv, save_private_test_csv
import torch
from torch.autograd import Variable
import numpy as np


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


def generate_prediction(model, out_path, hyper, evaluation=None):
    """
    Generate a prediction using <model> with <hyper> as hyperparameters
    Write the prediction to <out_path>. Optionally evaluate using <evaluation>.
    """
    train(model, **hyper)
    if evaluation is not None:
        acc = evaluate(model, **evaluation)
        print(f"Test accuracy: {acc}")

    private_test = load_private_test_csv('../data')
    predictions = []
    train_data = hyper['zero_train_data']
    for i, q in enumerate(private_test['question_id']):
        inputs = Variable(train_data[:, q]).unsqueeze(0)
        output = model(inputs, hyper['betas'][q])

        if output[0][private_test['user_id'][i]].item() >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)

    private_test['is_correct'] = predictions
    save_private_test_csv(private_test, out_path)
    return


if __name__ == '__main__':
    torch.manual_seed(2002)
    # Pretrain IRT model
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    thetas, betas, _, _, _ = irt(
        data=train_matrix.detach().numpy(),
        val_data=valid_data,
        lr=0.01,
        iterations=25
    )

    betas = (betas - 0.5) * 2

    ks = [10]
    lrs = [0.002]
    epochs = [30, 50, 100]

    for k in ks:
        for lr in lrs:
            for epoch in epochs:
                model = AutoEncoder(train_matrix.shape[0], k, 1)
                hyper = {
                    'lr': lr,
                    'lamb': 0.001,
                    'train_data': train_matrix,
                    'zero_train_data': zero_train_matrix,
                    'valid_data': valid_data,
                    'num_epoch': epoch,
                    'betas': betas,
                }
                evaluation = {
                    'train_data': zero_train_matrix,
                    'valid_data': test_data,
                    'betas': betas,
                }
                print(f"Starting: k={k}, lr={lr}, epoch={epoch}")
                generate_prediction(model, f'predictions/ae_mod_{k}_{lr}_{epoch}.csv', hyper, evaluation)

    # k = 50
    # model = AutoEncoder(train_matrix.shape[0], k, 1)
    # hyper = {
    #     'lr': 0.01,
    #     'lamb': 0.001,
    #     'train_data': train_matrix,
    #     'zero_train_data': zero_train_matrix,
    #     'valid_data': valid_data,
    #     'num_epoch': 15,
    #     'betas': betas,
    # }
    # evaluation = {
    #     'train_data': zero_train_matrix,
    #     'valid_data': test_data,
    #     'betas': betas,
    # }
    # generate_prediction(model, 'predictions/ae.csv', hyper, evaluation)
