"""
This module generates files that can be used to submit
to the CSC311 Kaggle competition.
"""
from part_a.item_response import irt
from part_b.ae_question_inject_by_mul import AutoEncoder, train, evaluate
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
        output = model(inputs, hyper['betas'][q], hyper['thetas'])

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

    thetas = torch.tensor(thetas, dtype=torch.float32)

    k = 10
    model = AutoEncoder(train_matrix.shape[0], k, 1)
    hyper = {
        'lr': 0.1,
        'lamb': 0.001,
        'train_data': train_matrix,
        'zero_train_data': zero_train_matrix,
        'valid_data': valid_data,
        'num_epoch': 10,
        'betas': betas,
        'thetas': thetas
    }
    evaluation = {
        'train_data': zero_train_matrix,
        'valid_data': test_data,
        'betas': betas,
        'thetas': thetas
    }
    generate_prediction(model, 'predictions/ae_question_inject_by_mul.csv', hyper, evaluation)
