from sklearn.impute import KNNImputer
from utils import *
import item_response as ir
import numpy as np
from neural_network import AutoEncoder, train
import torch
from torch.autograd import Variable


def bootstrapping(ori_dataset: dict, dataset_size):

    ori_dataset_idx = np.arange(len(ori_dataset['user_id']))
    iid_idx = np.random.choice(ori_dataset_idx,
                               size=(3, dataset_size)
                               )

    boots_datasets = []
    for i in range(3):
        dataset_i = {}
        for key in ori_dataset.keys():
            dataset_i[key] = np.array(ori_dataset[key])[iid_idx[i]].tolist()

        boots_datasets.append(dataset_i)

    return boots_datasets[0], boots_datasets[1], boots_datasets[2]



def dict_to_matrix(data: dict, full_shape: tuple) -> np.array:
    """
    Extract data from a dictionary to a numpy matrix, with shape <full_shape>
    data is in the form of {user_id: list, question_id: list, is_correct: list}
    """
    mat = np.empty(full_shape)
    mat[:] = np.nan
    for i in range(len(data['user_id'])):
        user_id = data['user_id'][i]
        question_id = data['question_id'][i]
        is_correct = data['is_correct'][i]
        mat[user_id, question_id] = is_correct
    return mat


def knn_train_predict(train_data: dict, test_data: dict, full_shape: tuple) -> list:
    """
    Use kNN to generate a prediction, hyperparameter as tuned in the previous part:
        - k = 11
        - Impute by user
    The input dictionaries are of form {user_id: list, question_id: list, is_correct: list}
    <full_shape> is the shape of the full train matrix
    """
    nbrs = KNNImputer(n_neighbors=11)

    # Obtain the matrix form of train data
    train_matrix = dict_to_matrix(train_data, full_shape)

    # Impute the matrix
    mat = nbrs.fit_transform(train_matrix)

    # Generate predictions
    predictions = []
    for i in range(len(test_data['user_id'])):
        user_id = test_data['user_id'][i]
        question_id = test_data['question_id'][i]
        predictions.append(mat[user_id, question_id])

    return predictions


def irt_train_test(train_data: dict, test_data: dict) -> list:
    """
    args:
    - train_data: ...
    - test_data: ...

    return:
    - preds: a list of prediction as probablity value of answering
             correctly, each has a value in [0, 1]
    """
    trained_theta, trained_beta, _, _, _ = \
                        ir.irt(data=train_data,
                            val_data=test_data, # dummpy, we don't use results from this
                            lr=0.01,
                            iterations=25
                            )

    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        x = (trained_theta[u] - trained_beta[q]).sum()
        p_a = ir.sigmoid(x)
        predictions.append(p_a >= 0.5)

    return predictions


def nn_train_predict(train_data: dict, full_shape: tuple, test_data: dict) -> list:
    """
    Use neural network to generate a prediction, hyperparameter as tuned in the previous part:
        - k = 100
        - learning rate = 0.1
        - epoch = 5
    The input dictionaries are of form {user_id: list, question_id: list, is_correct: list}
    """
    # Obtain the training matrix form of train data
    train_matrix = dict_to_matrix(train_data, full_shape)
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)
    # create neural network model and train it
    k = 10
    learning_rate = 0.1
    num_epoch = 10
    lamb = 0.001
    nn_model = AutoEncoder(train_matrix.shape[1], k)
    nn_model.train()
    train(nn_model, learning_rate, lamb, train_matrix, zero_train_matrix,
          test_data, num_epoch)

    # Make predictions
    nn_model.eval()
    predictions = []
    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        output = nn_model(inputs)
        guess = output[0][test_data["question_id"][i]].item() >= 0.5
        predictions.append(guess)
    return predictions


def ensemble_evaluate(pred1: list, pred2: list, pred3: list,
                      weight: tuple[int, int, int], test_data: dict) -> float:
    """
    Evaluate the ensemble of 3 models, whose predictions are given in
    pred1, 2, 3.
    """
    weight_normalized = [w / sum(weight) for w in weight]
    final_pred = [weight_normalized[0] * p1 +
                  weight_normalized[1] * p2 +
                  weight_normalized[2] * p3
                  for p1, p2, p3 in zip(pred1, pred2, pred3)]

    return evaluate(test_data, final_pred)


if __name__ == "__main__":
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_shape = max(train_data['user_id']) + 1, max(train_data['question_id']) + 1

    dataset_1, dataset_2, dataset_3\
                        = bootstrapping(
                            ori_dataset=train_data,
                            dataset_size=len(train_data['user_id'])
                            )

    prediction1 = knn_train_predict(dataset_1, val_data, train_shape)
    prediction2 = irt_train_test(dataset_2, val_data)
    prediction3 = nn_train_predict(dataset_3, train_shape, val_data)
    weight = (1, 2, 1)
    acc = ensemble_evaluate(prediction1, prediction2, prediction3, weight, val_data)
    print(f"Ensemble accuracy: {acc}")

    # valid_prediction1 = knn_train_predict(train_data, val_data, train_shape)
    # valid_prediction2 = irt_train_test(train_data, val_data)
    # valid_prediction3 = nn_train_predict(train_data, train_shape, val_data)
    # valid_acc = ensemble_evaluate(valid_prediction1, valid_prediction2, valid_prediction3, weight, val_data)
    # print(f"Ensemble accuracy on valid set: {valid_acc}")

    test_prediction1 = knn_train_predict(dataset_1, test_data, train_shape)
    test_prediction2 = irt_train_test(dataset_2, test_data)
    test_prediction3 = nn_train_predict(dataset_3, train_shape, test_data)
    test_acc = ensemble_evaluate(test_prediction1, test_prediction2, test_prediction3, weight, test_data)

    print(f"Ensemble accuracy on test set: {test_acc}")
