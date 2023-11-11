import torch
from torch import nn
import matplotlib.pyplot as plt
from utils import load_data, letter_to_tensor, sentence_to_tensor, LETTERS, get_random_sample
from pathlib import Path

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
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

MODEL_PATH = Path("./model.pt")
N_HIDDEN = 128
languages, language_names = load_data()
RNN_model = RNN(len(LETTERS), N_HIDDEN, len(languages))
criterion = nn.NLLLoss()
LR = 0.0005
optimizer = torch.optim.SGD(params=RNN_model.parameters(), lr=LR)


# Training setup
def train(name_tensor, language_tensor):
    next_hidden = RNN_model.init_hidden()

    for i in range(name_tensor.size()[0]):
        output, next_hidden = RNN_model(name_tensor[i], next_hidden)
    
    loss = criterion(output, language_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


### Model run
current_loss = 0.0
all_losses = []
STEPS = 1000
EPOCHS = 100000

# Load model if exists, else train
if MODEL_PATH.exists():
    RNN_model.load_state_dict(torch.load(MODEL_PATH))

for epoch in range(EPOCHS):
    language, language_tensor, name,  name_tensor = get_random_sample(languages, language_names)
    output, loss = train(name_tensor, language_tensor)
    current_loss += loss

    if epoch % STEPS == 0:
        all_losses.append(current_loss / STEPS)
        current_loss = 0

        prediction = languages[torch.argmax(output).item()]
        correct = "CORRECT" if prediction == language else f"WRONG -> {language}"
        print(f"{epoch} {epoch / EPOCHS * 100:.0f} {loss:.5f} {name} / {prediction} {correct}")

# Save the model to local disk
torch.save(RNN_model.state_dict(), MODEL_PATH)


# Plot losses per 1000 words
plt.plot(all_losses[1:-1], label='Loss Points')
# Adding titles and labels
plt.title('Loss Diagram')
plt.xlabel('Point in time')
plt.ylabel('Loss')
# Adding a legend
plt.legend()
# Display the plot
plt.show()

def predict(sentence):
    print(f"\n> {sentence}")
    with torch.inference_mode():
        name_tensor = sentence_to_tensor(sentence)

        next_hidden = RNN_model.init_hidden()
        
        for i in range(name_tensor.size()[0]):
            output, next_hidden = RNN_model(name_tensor[i], next_hidden)

        prediction = languages[torch.argmax(output).item()]
        print(prediction)

# Prediction state
while True:
    sentence = input("Input:")
    if sentence == "q":
        break
    else:
        predict(sentence)

