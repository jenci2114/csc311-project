from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    k_values = [1, 6, 11, 16, 21, 26]
    accs = []
    for k in k_values:
        acc_score = knn_impute_by_user(sparse_matrix, val_data, k)
        accs.append(acc_score)
    plt.plot(k_values, accs)
    plt.xlabel("k-Value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. k-Value")
    plt.savefig("../img/knn_impute_by_user.png", dpi=300)
    plt.show()

    for i in range(len(k_values)):
        print(f"k-value: {k_values[i]}, Accuracy: {accs[i]}")

    k_star_index = np.argmax(accs)
    k_star = k_values[k_star_index]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print(f"k_star: {k_star}, Test Accuracy: {test_acc}")


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
