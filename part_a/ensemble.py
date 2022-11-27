import numpy as np
from item_response import irt, sigmoid

def bootstrapping(dataset, k):
    
    
    return 

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
                        irt(data=train_data, 
                            val_data=test_data, # dummpy, we don't use results from this
                            lr=0.01, 
                            iterations=1000
                            )
                        
    preds = []
    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        x = (trained_theta[u] - trained_beta[q]).sum()
        p_a = sigmoid(x)
        preds.append(p_a >= 0.5)
    
    return preds
    
    
    