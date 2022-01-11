import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
vec =[float(x) for x in (input("Enter a vector: like [1,-2,0.5]")).split(",")]
print(softmax(vec))
