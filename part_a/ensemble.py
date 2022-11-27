import numpy as np
from sklearn.impute import KNNImputer
from utils import evaluate


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
    train_matrix = np.empty(full_shape)
    train_matrix[:] = np.nan
    for i in range(len(train_data['user_id'])):
        user_id = train_data['user_id'][i]
        question_id = train_data['question_id'][i]
        is_correct = train_data['is_correct'][i]
        train_matrix[user_id, question_id] = is_correct

    # Impute the matrix
    mat = nbrs.fit_transform(train_matrix)

    # Generate predictions
    predictions = []
    for i in range(len(test_data['user_id'])):
        user_id = test_data['user_id'][i]
        question_id = test_data['question_id'][i]
        predictions.append(mat[user_id, question_id])

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
