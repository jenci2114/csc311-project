from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

from torch import sigmoid
from part_a import item_response

from torch.utils.data import DataLoader, Dataset
from meta_process import process_question_meta, get_subject_number


def read_encoded_question_metadata(filepath, question_num, k):
    """Read the encoded question metadata from <filepath>."""
    res = np.zeros((question_num, k))
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            q_id = int(row[0])
            res[q_id] = np.array(row[1:]).astype(float)
    return res

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (prior_train_matrix, train_data, valid_data, test_data)
        WHERE:
        prior_train_matrix: 2D sparse matrix where missing entries are
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

    prior_train_matrix = train_matrix.copy()
    for q in range(train_matrix.shape[1]):
        curr_mean = np.nanmean(train_matrix[:, q])
        nan_mask = np.isnan(train_matrix[:, q])
        prior_train_matrix[nan_mask, q] = curr_mean
    prior_train_matrix = torch.FloatTensor(prior_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return prior_train_matrix, train_matrix, valid_data, test_data


class AEDataset(Dataset):
    def __init__(self, prior_train_matrix, beta_vector, metadata) -> None:
        """
        args:
            - zero_train_matrix: table with nan replaced by 0.
        """
        super().__init__()
        self.prior_train_matrix = prior_train_matrix
        self.beta_vector = beta_vector
        self.metadata = metadata


    def __len__(self):
        return self.prior_train_matrix.shape[1]

    def __getitem__(self, idx):
        """
        args:
            - idx: question idx
        """
        return {'question_id': idx,
                'question_vector': self.prior_train_matrix[:, idx],
                'beta': torch.tensor([self.beta_vector[idx]], dtype=torch.float32)
                        if self.beta_vector is not None else torch.nan,
                'meta_latent': self.metadata[idx]
                        if self.metadata is not None else torch.nan
                }


class AutoEncoder(nn.Module):
    def __init__(self, num_students, num_subjects, question_latent_dim=10, subject_latent_dim=5, extra_latent_dim=0):
        """ Initialize a class AutoEncoder."""
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_students, question_latent_dim)
        self.h = nn.Linear(question_latent_dim + subject_latent_dim + extra_latent_dim, num_students)

        self.subject_enc_linear = nn.Linear(num_subjects, subject_latent_dim)

    def get_weight_norm(self):
        """ Return the weight norm"""
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        subject_w_norm = torch.norm(self.subject_enc_linear.weight, 2) ** 2
        return g_w_norm + h_w_norm + subject_w_norm

    def get_raw_latent(self, inputs):
        return sigmoid(self.g(inputs))

    def forward(self, inputs, beta=None, meta=None):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        question_raw_latent = sigmoid(self.g(inputs))
        if beta is not None:
            question_latent = torch.cat((question_raw_latent, beta), axis=-1)
        else:
            question_latent = question_raw_latent

        if meta is not None:
            subject_latent = sigmoid(self.subject_enc_linear(meta))
            question_full_latent = torch.cat(
                (question_latent, subject_latent),
                axis=-1
            )
        else:
            question_full_latent = question_latent

        decoded = sigmoid(self.h(question_full_latent))
        return decoded


def train(
    model,
    lr,
    batch_size,
    lamb,
    train_data,
    prior_train_data,
    valid_data,
    num_epoch,
    betas,
    metadata
    ):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param prior_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Build dataset object
    dataset = AEDataset(
        prior_train_matrix=prior_train_data,
        beta_vector=betas,
        metadata=metadata
        )

    # Define dataloader
    dataloarder = DataLoader(
                            dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True
                            )

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for datapoints in dataloarder:
            question_id_batch = datapoints['question_id']
            question_vectors_batch = datapoints['question_vector']
            beta_batch = datapoints['beta']
            meta_batch = datapoints['meta_latent']

            inputs = Variable(question_vectors_batch)
            targets = inputs.clone()

            optimizer.zero_grad()

            outputs = model(
                        inputs,
                        beta=beta_batch if not torch.isnan(beta_batch.flatten()[0]).item() else None,
                        meta=meta_batch if not torch.isnan(meta_batch.flatten()[0]).item() else None,
                        )
            nan_mask = np.isnan(train_data[:, question_id_batch].numpy()).T
            nan_mask = torch.tensor(nan_mask)


            targets[nan_mask] = outputs[nan_mask]

            regularizer = 0.5 * lamb * model.get_weight_norm()
            loss = torch.sum((outputs - targets) ** 2.) + regularizer
            loss = torch.sum((outputs - targets) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, prior_train_data, valid_data, betas, metadata)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    return model


def evaluate(model, train_data, valid_data, betas, metadata):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, q in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[:, q]).unsqueeze(0)
        output = model(
            inputs=inputs,
            beta=torch.tensor([[betas[q]]], dtype=torch.float32) if betas is not None else None,
            meta=metadata[q].unsqueeze(0) if metadata is not None else None
        )


        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def get_latent_mat(model, zero_train_data, entity='question'):
    if entity == 'question':
        batched_input = torch.t(zero_train_data)
        batched_latent = model.get_raw_latent(batched_input)
        latent_mat = torch.t(batched_latent)
        breakpoint()
        return latent_mat.detach().numpy()




def main():
    prior_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Pre-train IRT model
    _, betas, _, _, _ = \
        item_response.irt(
        data=train_matrix.detach().numpy(),
        val_data=valid_data,
        lr=0.01,
        iterations=25,
    )

    # Load raw question metadata
    data = load_train_csv('../data')
    question_count = max(data['question_id']) + 1
    subject_count = get_subject_number('../data/subject_meta.csv')
    raw_question_meta = process_question_meta('../data/question_meta.csv', question_count, subject_count)
    raw_question_meta = torch.FloatTensor(raw_question_meta)


    # betas = None
    batch_sizes = [5, 10, 30]

    # Set model hyperparameters.
    meta_latent_dim_list = [2, 3, 4, 5]
    k_list = [3, 5, 8, 10]
    lr_list = [1e-2, 1e-3]
    epoch_list = [3, 5, 10, 15, 20, 25]
    for batch_size in batch_sizes:
        for k in k_list:
            for meta_latent_dim in meta_latent_dim_list:
                for lr in lr_list:
                    for num_epoch in epoch_list:
                        extra_latent_dim = 1 if betas is not None else 0
                        model = AutoEncoder(
                            train_matrix.shape[0],
                            subject_count,
                            k,
                            subject_latent_dim=meta_latent_dim,
                            extra_latent_dim=extra_latent_dim)

                        print(f"Start training model with k={k}, k_meta={meta_latent_dim}, bs={batch_size}, lr={lr}, epochs={num_epoch}")
                        train(
                            model,
                            lr,
                            batch_size,
                            0,
                            train_matrix,
                            prior_train_matrix,
                            valid_data,
                            num_epoch,
                            betas,
                            raw_question_meta
                            )
                        print("")


if __name__ == "__main__":
    torch.manual_seed(2002)
    main()
