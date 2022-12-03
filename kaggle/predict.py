"""
This module generates files that can be used to submit
to the CSC311 Kaggle competition.
"""
from part_a.item_response import irt
from part_b.ae_question_inject_meta import AutoEncoder, train, evaluate, read_encoded_question_metadata
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

        if 'betas' in hyper:
            beta = hyper['betas'][q]
        else:
            beta = None

        if 'metas' in hyper:
            meta = hyper['metas'][q]
        else:
            meta = None

        output = model(inputs, beta=beta, meta=meta)

        if output[0][private_test['user_id'][i]].item() >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)

    private_test['is_correct'] = predictions
    save_private_test_csv(private_test, out_path)
    return


if __name__ == '__main__':
    torch.manual_seed(311412)
    # Pretrain IRT model
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    thetas, betas, _, _, _ = irt(
        data=train_matrix.detach().numpy(),
        val_data=valid_data,
        lr=0.01,
        iterations=25
    )

    betas = (betas - 0.5) * 2

    question_num = train_matrix.shape[1]
    k_meta = 5
    metadata = read_encoded_question_metadata('../data/question_meta_encoded.csv', question_num, k_meta)

    ks = [10, 20, 50]
    lrs = [0.01]
    epochs = [5, 10, 15, 20]

    for k in ks:
        for lr in lrs:
            for epoch in epochs:
                model = AutoEncoder(train_matrix.shape[0], k, 1 + k_meta)
                hyper = {
                    'lr': lr,
                    'lamb': 0.001,
                    'train_data': train_matrix,
                    'zero_train_data': zero_train_matrix,
                    'valid_data': valid_data,
                    'num_epoch': epoch,
                    'betas': betas,
                    'metas': metadata
                }
                evaluation = {
                    'train_data': zero_train_matrix,
                    'valid_data': test_data,
                    'betas': betas,
                    'metas': metadata
                }
                print(f"Starting: k={k}, lr={lr}, epoch={epoch}")
                generate_prediction(model, f'predictions/ae_meta_{k}_{lr}_{epoch}.csv', hyper, evaluation)

    print("Done")
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
