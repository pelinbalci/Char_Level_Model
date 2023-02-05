import copy
import numpy as np

""""
Exploding gradients
When gradients are very large, they're called "exploding gradients."
Exploding gradients make the training process more difficult, because the updates may be so large that they "overshoot" 
the optimal values during back propagation.

Very large, or "exploding" gradients updates can be so large that they "overshoot" the optimal values during back prop 
-- making training difficult
Clip gradients before updating the parameters to avoid exploding gradients
"""


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''
    gradients = copy.deepcopy(gradients)

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    ### START CODE HERE ###
    # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in dWaa, dWax, dWya, db, dby:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    ### END CODE HERE ###

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


np.random.seed(3)
dWax = np.round(np.random.randn(5, 3),2) * 10
dWaa = np.round(np.random.randn(5, 5),2) * 10
dWya = np.round(np.random.randn(2, 5),2) * 10
db = np.round(np.random.randn(5, 1),2) * 10
dby = np.round(np.random.randn(2, 1),2) * 10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients2 = clip(gradients, 10)

print(gradients)
print("clipped")
print(gradients2)