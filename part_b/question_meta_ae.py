import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch import sigmoid
import numpy as np
from meta_process import process_question_meta, get_subject_number
from utils import load_train_csv
import csv


class QuestionMetaAE(nn.Module):
    def __init__(self, num_subjects, k=2):
        super(QuestionMetaAE, self).__init__()

        self.g = nn.Linear(num_subjects, k)
        self.h = nn.Linear(k, num_subjects)

    def forward(self, inputs):
        encoded = sigmoid(self.g(inputs))
        decoded = sigmoid(self.h(encoded))
        return decoded

    def latent(self, inputs):
        encoded = sigmoid(self.g(inputs))
        return encoded


def train(model, data, lr, num_epoch):
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_question = data.shape[0]

    for epoch in range(num_epoch):
        train_loss = 0.

        for question_id in range(num_question):
            inputs = Variable(data[question_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {} | Loss: {:.6f}'.format(epoch, train_loss / num_question))

    return model


def extract_latent(model, data, k):
    """
    Extract the latent values for each question.
    Return a numpy array with shape(num_question, k) where
    res[i][j] is the j-th latent value of the i-th question.
    """
    model.eval()
    num_question = data.shape[0]
    res = np.zeros((num_question, k))
    for question_id in range(num_question):
        inputs = Variable(data[question_id]).unsqueeze(0)
        latent = model.latent(inputs)
        res[question_id] = latent.data.numpy()
    return res


def assess_result(model, data, k):
    """
    Assess the result of the model by computing the reconstruction.
    Return a numpy array with shape(num_question, num_subjects) where
    res[i][j] is the j-th subject value of the i-th question
    """
    model.eval()
    num_question = data.shape[0]
    num_subjects = data.shape[1]
    res = np.zeros((num_question, num_subjects))
    for question_id in range(num_question):
        inputs = Variable(data[question_id]).unsqueeze(0)
        reconstructed = model(inputs)
        res[question_id] = reconstructed.data.numpy()
    return res


def latent_to_csv(latent, filename):
    """Save the latent values to the specified file."""
    k = latent.shape[1]
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['question_id'] + [f'latent{i}' for i in range(k)])
        for i in range(latent.shape[0]):
            writer.writerow([i] + list(latent[i]))
    return


if __name__ == '__main__':
    data = load_train_csv('../data')
    question_count = max(data['question_id']) + 1
    subject_count = get_subject_number('../data/subject_meta.csv')
    question_meta = process_question_meta('../data/question_meta.csv', question_count, subject_count)
    question_meta = torch.FloatTensor(question_meta)

    k = 5
    lr = 0.1
    num_epoch = 1500
    model = QuestionMetaAE(subject_count, k)
    model = train(model, question_meta, lr, num_epoch)
    latent_meta = extract_latent(model, question_meta, k)
    reconstructed_meta = assess_result(model, question_meta, subject_count)
