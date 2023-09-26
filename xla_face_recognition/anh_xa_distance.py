import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

score = 0.8
distance = score # Khoảng cách giữa hai vector embedding
similarity_percentage = sigmoid(distance) * 100
print(similarity_percentage)