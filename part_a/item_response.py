from utils import *

import numpy as np
import math
from matplotlib import pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # (N_student, N_questions)-shaped matrix, where
    # the (i,j)-th entry = (theta_i - beta_j)
    offset = theta[:, np.newaxis] - beta[np.newaxis, :]   
    
    if isinstance(data, np.ndarray):
        # Set up the matrix to sum over
        matrix_to_sum = data * offset \
                        - np.log(1 + np.exp(offset))
        
        # Sum the whole matrix, treating nan as 0
        log_lklihood = np.nansum(matrix_to_sum)
    elif isinstance(data, dict):
        # log_lklihood = 0
        # for i in range(len(data["is_correct"])):
        #     cur_user_id = data["user_id"][i]
        #     cur_question_id = data["question_id"][i]
        #     offset_entry = offset[cur_user_id, cur_question_id]
        #     answer_correct = data["is_correct"][i]  # c_ij, 0 or 1
        #     log_lklihood += (answer_correct * offset_entry) \
        #                     - math.log(1 + math.exp(offset_entry))
        
        u_entries = data["user_id"]
        q_entries = data["question_id"]
        offset_entries = offset[u_entries, q_entries]
        answer_correct = data["is_correct"]
        array_to_sum = (answer_correct * offset_entries) \
                        - np.log(1 + np.exp(offset_entries))
        log_lklihood = array_to_sum.sum()

    else:
        raise Exception(f"Argument <data> has to be one of types \
                        np.ndarray or dict, but {type(data)} is received")
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################    
    # (N_student, N_questions)-shaped matrix, where
    # the (i,j)-th entry = (theta_i - beta_j)
    offset = theta[:, np.newaxis] - beta[np.newaxis, :]   
    
    # Set up the matrix to sum over
    mat_to_sum = data - sigmoid(offset)
    
    # Calculate the derivative (vector), using nan-sum (treat nan as 0)
    d_theta = np.nansum(mat_to_sum, axis=1)
    d_beta = np.nansum((-1) * mat_to_sum, axis=0)
    
    # Max the log-lklihood so *add* derivative (gradient ascent)
    theta = theta + lr * d_theta  
    beta = beta + lr * d_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(data.shape[0])
    beta = np.zeros(data.shape[1])
    
    theta_lst = []  # Added line: A list of len = iterations
    beta_lst = []    # Added line: A list of len = iterations

    val_acc_lst = []

    for i in range(iterations):
        theta_lst.append(theta)   # Added line: augement the list
        beta_lst.append(beta)     # Added line: augement the list
        
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, theta_lst, beta_lst  


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Hyperparameters
    lr = 0.01
    iterations = 25
    
    
    # Training & Logging 
    trained_theta, trained_beta, val_acc_lst, theta_lst, beta_lst = \
                            irt(data=sparse_matrix.toarray(),  # Scipy sparse matrix --> numpy array
                                val_data=val_data,
                                lr=lr,
                                iterations=iterations
                                )
                         
                            
    # Plot for 2b, using logged parameters
    train_LLKs = []
    val_LLKs = []
    for i in range(iterations):
        train_LLKs.append( 
            (-1) * 
            neg_log_likelihood(
                data=sparse_matrix.toarray(),
                theta=theta_lst[i],
                beta=beta_lst[i]
            )
        )
        val_LLKs.append(
            (-1) * 
            neg_log_likelihood(
                data=val_data,
                theta=theta_lst[i],
                beta=beta_lst[i]
            )
        )

    xs = np.arange(iterations)
    
    plt.plot(xs, train_LLKs)
    plt.xlabel("Iterations")
    plt.ylabel("Log-likelihood")
    plt.title("(Train) Iterations v.s. Log-likelihood")
    plt.tight_layout()
    plt.savefig("../(2b)train_LLK.pdf")
    plt.clf()
    
    plt.plot(xs, val_LLKs)
    plt.xlabel("Iterations")
    plt.ylabel("Log-likelihood")
    plt.title("(Validate) Iterations v.s. Log-likelihood")
    plt.tight_layout()
    plt.savefig("../(2b)val_LLK.pdf")
    plt.clf()
    
    
    # Report accuracy for 2c
    final_val_acc = evaluate(data=val_data, theta=trained_theta, beta=trained_beta)
    final_test_acc = evaluate(data=test_data, theta=trained_theta, beta=trained_beta)
    print("The final val accuracy: ", final_val_acc)
    print("The final test accuracy: ", final_test_acc)
    
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1 = 10
    j2 = 20
    j3 = 200
    thetas = np.linspace(-5, 5, 500)
    prob_correct_j1 = sigmoid(thetas - trained_beta[j1])
    prob_correct_j2 = sigmoid(thetas - trained_beta[j2])
    prob_correct_j3 = sigmoid(thetas - trained_beta[j3])
    plt.plot(thetas, prob_correct_j1, label=f"Question $j = j_1$")
    plt.plot(thetas, prob_correct_j2, label="Question $j = j_2$")
    plt.plot(thetas, prob_correct_j3, label="Question $j = j_3$")
    plt.xlabel("$\\theta$")
    plt.ylabel("Probability $p(c_{ij} = 1 | \\theta, \\beta_{j})$")
    plt.legend(loc='lower right')
    plt.title("$\\theta$  v.s.  Probability $p(c_{ij} = 1 | \\theta, \\beta_{j})$")
    plt.savefig("../(2d).pdf")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
