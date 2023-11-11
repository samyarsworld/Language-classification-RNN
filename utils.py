import torch
import unicodedata
from glob import glob
import os
import io


# Letters in out vocabulary
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"

# Stores letters as tensor in one-hot encoding format
def letter_to_tensor(letter, LETTERS=LETTERS):
    t = torch.zeros(len(LETTERS))
    t[LETTERS.index(letter)] = 1
    t = t.unsqueeze(dim=0)
    return t

letter_to_tensor("a")
# Returns tensor format of a line of letters
def line_to_tensor(word):
    t = torch.tensor([])
    return (torch.cat(list(letter_to_tensor[letter] for letter in word), dim=0)).unsqueeze(1)


# Turns unicode to asci
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in s
    )

# Loads data into categories (classes) and data in the categories
def load_data():
    class_data = {}
    classes = []
    files = glob("data/names/*.txt")

    for file in files:
        name, ex = os.path.splitext(os.path.basename(file))
        classes.append(name)

        names = io.open(file, encoding="utf-8").read().strip().split("\n")
        names = [unicode_to_ascii(name) for name in names]

        class_data[name] = names

    return classes, class_data