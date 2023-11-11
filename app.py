import torch
from torch import nn
import matplotlib.pyplot as plt
from utils import load_data, letter_to_tensor, LETTERS



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.i2o = nn.Linear(in_features=input_size + hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined_tensor = torch.cat((x, h), dim=1)
        return self.softmax(self.i2o(combined_tensor)), self.i2h(combined_tensor)
    


classes, class_names = load_data()
n_hidden = 128
RNN_model = RNN(len(LETTERS), n_hidden, len(classes))

output, next_hidden = RNN_model(letter_to_tensor("A"), torch.zeros(1, n_hidden))

print(letter_to_tensor("A").shape)
print(output.shape)
print(next_hidden.shape)
