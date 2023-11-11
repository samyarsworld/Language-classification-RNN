import torch
import unicodedata
from glob import glob
import os
import io
import random

# Letters in out vocabulary
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"

# Stores letters as tensor in one-hot encoding format
def letter_to_tensor(letter, LETTERS=LETTERS):
    t = torch.zeros(len(LETTERS))
    t[LETTERS.index(letter)] = 1
    t = t.unsqueeze(dim=0)
    return t

# Returns tensor format of a line of letters
def sentence_to_tensor(sentence):
    t = torch.tensor([])
    return (torch.cat(list(letter_to_tensor(letter) for letter in sentence), dim=0)).unsqueeze(1)


# Turns unicode to asci
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in LETTERS
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


def get_random_sample(languages, language_names):
    language = random.choice(languages)
    random_name = random.choice(language_names[language])
    random_name_tensor = sentence_to_tensor(random_name)
    language_tensor = torch.tensor([languages.index(language)], dtype=torch.long)

    return language, language_tensor, random_name,  random_name_tensor
    



