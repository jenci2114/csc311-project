"""
This module generates files that can be used to submit
to the CSC311 Kaggle competition.
"""
from part_a.item_response import irt
from part_b.ae_best_epoch import AutoEncoder, train, evaluate, read_encoded_question_metadata, load_data
from utils import load_train_sparse, load_valid_csv, load_public_test_csv, \
    load_private_test_csv, save_private_test_csv
import torch
from torch.autograd import Variable
import numpy as np


def generate_prediction(model, out_path, hyper, evaluation=None):
    """
    Generate a prediction using <model> with <hyper> as hyperparameters
    Write the prediction to <out_path>. Optionally evaluate using <evaluation>.
    """
    best_model = train(model, **hyper)
    if evaluation is not None:
        acc = evaluate(best_model, **evaluation)
        print(f"Test accuracy: {acc}")

    private_test = load_private_test_csv('../data')
    predictions = []
    train_data = hyper['prior_train_data']
    for i, q in enumerate(private_test['question_id']):
        inputs = Variable(train_data[:, q]).unsqueeze(0)

        if 'betas' in hyper:
            beta = torch.tensor([[hyper['betas'][q]]], dtype=torch.float32)
        else:
            beta = None

        if 'metadata' in hyper:
            meta = hyper['metadata'][q].unsqueeze(0)
        else:
            meta = None

        output = best_model(inputs, beta=beta, meta=meta)

        if output[0][private_test['user_id'][i]].item() >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)

    private_test['is_correct'] = predictions
    save_private_test_csv(private_test, out_path)
    return


if __name__ == '__main__':
    torch.manual_seed(31415926)
    # Pretrain IRT model
    prior_train_matrix, train_matrix, valid_data, test_data = load_data()
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
    metadata = torch.tensor(metadata, dtype=torch.float32)

    ks = [5, 6, 8, 10, 12, 15, 20, 30]
    lrs = [0.001]
    epochs = [10, 15, 20]
    batch_sizes = [5, 10, 15, 20]

    for k in ks:
        for lr in lrs:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    model = AutoEncoder(train_matrix.shape[0], k, 1 + k_meta)
                    hyper = {
                        'lr': lr,
                        'lamb': 0.0,
                        'batch_size': batch_size,
                        'train_data': train_matrix,
                        'prior_train_data': prior_train_matrix,
                        'valid_data': valid_data,
                        'num_epoch': epoch,
                        'betas': betas,
                        'metadata': metadata
                    }
                    evaluation = {
                        'train_data': prior_train_matrix,
                        'valid_data': test_data,
                        'betas': betas,
                        'metadata': metadata
                    }
                    print(f"Starting: k={k}, lr={lr}, epoch={epoch}, batch_size={batch_size}")
                    generate_prediction(model, f'predictions/ae_adam_{k}_{lr}_{epoch}_{batch_size}.csv', hyper, evaluation)

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
