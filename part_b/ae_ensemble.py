import ae
import ae_user
from part_a import item_response
from ae import AutoEncoder
from ae_user import AutoEncoderUser
from utils import *
import torch
from torch.autograd import Variable


def bootstrapping(ori_dataset: dict, dataset_size):
    ori_dataset_idx = np.arange(len(ori_dataset['user_id']))
    iid_idx = np.random.choice(ori_dataset_idx,
                               size=(2, dataset_size)
                               )

    boots_datasets = []
    for i in range(2):
        dataset_i = {}
        for key in ori_dataset.keys():
            dataset_i[key] = np.array(ori_dataset[key])[iid_idx[i]].tolist()

        boots_datasets.append(dataset_i)

    return boots_datasets[0], boots_datasets[1]


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


def ae_user_predict(zero_train_matrix, train_matrix, test_data, thetas=None) -> list:
    """
    Use user-based autoencoder to generate a prediction, hyperparameter as following:
        - k = 20
        - learning rate = 0.01
        - epoch = 20
    """
    k = 20
    lr = 0.01
    num_epoch = 20
    if thetas is not None:
        model = AutoEncoderUser(train_matrix.shape[1], k, 1)
    else:
        model = AutoEncoderUser(train_matrix.shape[1], k)
    ae_user.train(model, lr, None, train_matrix, zero_train_matrix, test_data, num_epoch, thetas)

    # Predict
    model.eval()
    predictions = []
    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        if thetas is not None:
            output = model(inputs, thetas[u])
        else:
            output = model(inputs)
        predictions.append(output[0][test_data["question_id"][i]].item())
    return predictions


def ae_question_predict(zero_train_matrix, train_matrix, test_data, betas=None) -> list:
    """
    Use question-based autoencoder to generate a prediction, hyperparameter as following:
        - k = 20
        - learning rate = 0.01
        - epoch = 15
    """
    k = 20
    lr = 0.01
    num_epoch = 15
    if betas is not None:
        model = AutoEncoder(train_matrix.shape[0], k, 1)
    else:
        model = AutoEncoder(train_matrix.shape[0], k)
    ae.train(model, lr, None, train_matrix, zero_train_matrix, test_data, num_epoch, betas)

    # Predict
    model.eval()
    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        inputs = Variable(zero_train_matrix[:, q]).unsqueeze(0)
        if betas is not None:
            output = model(inputs, betas[q])
        else:
            output = model(inputs)
        predictions.append(output[0][test_data["user_id"][i]].item())
    return predictions


def ensemble_evaluate(pred1: list, pred2: list, weight: tuple[int, int], test_data: dict) -> float:
    """
    Evaluate the ensemble of 2 models, whose predictions are given in pred1, 2
    """
    weight_normalized = [w / sum(weight) for w in weight]
    final_pred = [weight_normalized[0] * p1 +
                  weight_normalized[1] * p2
                  for p1, p2 in zip(pred1, pred2)]
    return evaluate(test_data, final_pred)


if __name__ == '__main__':
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_shape = max(train_data['user_id']) + 1, max(train_data['question_id']) + 1

    # Pretrain IRT model
    thetas, betas, _, _, _ = item_response.irt(
        data=train_data,
        val_data=test_data,
        lr=0.01,
        iterations=25
    )

    # Bootstrap
    dataset_1, dataset_2 = bootstrapping(train_data, len(train_data['user_id']))

    train_matrix_1 = dict_to_matrix(dataset_1, train_shape)
    zero_train_matrix_1 = train_matrix_1.copy()
    zero_train_matrix_1[np.isnan(train_matrix_1)] = 0
    zero_train_matrix_1 = torch.FloatTensor(zero_train_matrix_1)
    train_matrix_1 = torch.FloatTensor(train_matrix_1)

    train_matrix_2 = dict_to_matrix(dataset_2, train_shape)
    zero_train_matrix_2 = train_matrix_2.copy()
    zero_train_matrix_2[np.isnan(train_matrix_2)] = 0
    zero_train_matrix_2 = torch.FloatTensor(zero_train_matrix_2)
    train_matrix_2 = torch.FloatTensor(train_matrix_2)

    # Prediction from user-based autoencoder
    predictions_1 = ae_user_predict(zero_train_matrix_1, train_matrix_1, val_data, thetas)

    # Prediction from question-based autoencoder
    predictions_2 = ae_question_predict(zero_train_matrix_2, train_matrix_2, val_data, betas)

    # Combine predictions
    weight = (1, 2)
    acc = ensemble_evaluate(predictions_1, predictions_2, weight, val_data)
    print(f"Validation accuracy: {acc}")
