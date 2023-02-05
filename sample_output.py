import numpy as np
from rnn_helper import softmax

"""
Sampling is a technique you can use to pick the index of the next character according to a probability distribution.
To begin character-level sampling:
Input a "dummy" vector of zeros as a default input
Run one step of forward propagation to get 𝑎⟨1⟩ (your first character) and 𝑦̂ ⟨1⟩ (probability distribution for the 
following character)
When sampling, avoid generating the same result each time given the starting letter (and make your names more 
interesting!) by using np.random.choice
"""


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- Python dictionary mapping each character to an index.
    seed -- Used for grading purposes. Do not worry about it.

    Returns:
    indices -- A list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    ### START CODE HERE ###
    # Step 1: Create the a zero vector x that can be used as the one-hot vector
    # Representing the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []

    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    # Initialize idx to -1
    idx = -1

    # Loop over time-steps t. At each time-step:
    # Sample a character from a probability distribution
    # And append its index (`idx`) to the list "indices".
    # You'll stop if you reach 50 characters
    # (which should be very unlikely with a well-trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops.
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # For grading purposes
        np.random.seed(counter + seed)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        # (see additional hints above)
        idx = np.random.choice(range(len(y)), p=y.reshape([1, -1])[0])

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        # (see additional hints above)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1
        counter += 1

    ### END CODE HERE ###

    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices


data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

seed=24
np.random.seed(24)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
vocab_size = by.shape[0]
n_a = Waa.shape[1]

print("vocab_size", vocab_size, "n_a", n_a)

x = np.zeros((vocab_size,1))
a_prev = np.zeros((n_a,1))


lin_reg_output = np.dot(Wax, x) + np.dot(Waa, a_prev) + b
a = np.tanh(lin_reg_output)
z_lin_reg_output = np.dot(Wya, a) + by
y = softmax(z_lin_reg_output)
print(len(y)) #output for each 27 vocabulary

max_prob_selection_idx = [i for i in range(len(y)) if y[i]==y.max()][0]
max_prob_selection_char = ix_to_char[max_prob_selection_idx]
print(max_prob_selection_idx, max_prob_selection_char)

random_prob_selection_idx = np.random.choice(range(len(y)), p=y.reshape([1,-1])[0])
random_prob_selection_char = ix_to_char[random_prob_selection_idx]
print(random_prob_selection_idx, random_prob_selection_char)

indices = []
idx_ = np.random.choice(range(len(y)), p=y.reshape([1,-1])[0])
indices.append(idx_)
print(indices)