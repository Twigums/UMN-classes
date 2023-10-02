import numpy as np
import matplotlib.pyplot as plt
import csv

with open("hw1_q5_dataset.csv", newline = "\n") as file:
    data = csv.reader(file, delimiter = " ")

    for row in data:
        print(row)
