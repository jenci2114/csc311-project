import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch import sigmoid
from meta_process import process_question_meta, get_subject_number
from utils import load_train_csv


class QuestionMetaAE(nn.Module):
    def __init__(self, num_subjects, k=2):
        super(QuestionMetaAE, self).__init__()

        self.g = nn.Linear(num_subjects, k)
        self.h = nn.Linear(k, num_subjects)

    def forward(self, inputs):
        encoded = sigmoid(self.g(inputs))
        decoded = sigmoid(self.h(encoded))
        return decoded


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


if __name__ == '__main__':
    data = load_train_csv('../data')
    question_count = max(data['question_id']) + 1
    subject_count = get_subject_number('../data/subject_meta.csv')
    question_meta = process_question_meta('../data/question_meta.csv', question_count, subject_count)
    question_meta = torch.FloatTensor(question_meta)

    k = 2
    lr = 0.01
    num_epoch = 15
    model = QuestionMetaAE(subject_count, k)
    model = train(model, question_meta, lr, num_epoch)
