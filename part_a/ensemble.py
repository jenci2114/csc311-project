from sklearn.impute import KNNImputer
from utils import *
import item_response as ir
import numpy as np


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
                            iterations=1000
                            )

    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        x = (trained_theta[u] - trained_beta[q]).sum()
        p_a = ir.sigmoid(x)
        predictions.append(p_a >= 0.5)

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


    dataset_1, dataset_2, dataset_3\
                        = bootstrapping(
                            ori_dataset=train_data,
                            dataset_size=len(train_data['user_id'])
                            )

    
    predictions_1 = irt_train_test(train_data=dataset_1,
                                    test_data=test_data)
    predictions_2 = irt_train_test(train_data=dataset_2,
                                    test_data=test_data)
    predictions_3 = irt_train_test(train_data=dataset_3,
                                    test_data=test_data)
    
    acc = ensemble_evaluate(predictions_1, pred2=predictions_2, pred3=predictions_3)
    print(acc)
