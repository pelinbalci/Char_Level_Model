import numpy as np
from rnn_helper import *
from helper_functions import *
from clip_gradient import *
from sample_output import *

data = open('dinos.txt', 'r').read()
data = data.lower()
data_x = data.split("\n")
names = [x.strip() for x in data.split("\n")]
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
num_iterations = 22001
n_a = 50  # hidden layers
dino_names = 7  # define 7 new names  - seq_length
vocab_size = 27
verbose = True
learning_rate = 0.01


char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}


# Retrieve n_x and n_y from vocab_size
n_x, n_y = vocab_size, vocab_size
# Initialize parameters
np.random.seed(1)
Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
b = np.zeros((n_a, 1))  # hidden bias
by = np.zeros((n_y, 1))  # output bias
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
# Initialize loss (this is required because we want to smooth our loss)
loss = -np.log(1.0/vocab_size)*dino_names
# Build list of all dinosaur names (training examples).
examples = [x.strip() for x in data_x]
# Shuffle list of all dinosaur names
np.random.seed(0)
np.random.shuffle(examples)
# Initialize the hidden state of your LSTM
a_prev = np.zeros((n_a, 1))
# for grading purposes
last_dino_name = "abc"
# Optimization loop
for j in range(num_iterations):
    # Set the index `idx` (see instructions above)
    idx = j % len(examples)
    # Set the input X (see instructions above) use a dinasour's name as input
    single_example = examples[idx]
    single_example_chars = [char for char in single_example]
    single_example_ix = [char_to_ix[c] for c in single_example_chars]
    X = [None] + [single_example_ix]
    X = [None] + [char_to_ix[ch] for ch in examples[idx]]

    # Set the labels Y (see instructions above)
    ix_newline = char_to_ix["\n"]
    Y = X[1:] + [ix_newline]

    # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
    # Choose a learning rate of 0.01
    # Forward propagate through time (≈1 line)   cache=(y_hat, a, x)
    # curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)

    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    a_prev = a[len(X) - 1]
    curr_loss = loss

    # debug statements to aid in correctly forming X, Y
    if verbose and j in [0, len(examples) - 1, len(examples)]:
        print("j = ", j, "idx = ", idx, )
    if verbose and j in [0]:
        print("single_example =", single_example)
        print("single_example_chars", single_example_chars)
        print("single_example_ix", single_example_ix)
        print(" X = ", X, "\n", "Y =       ", Y, "\n")

    # to keep the loss smooth.
    loss = smooth(loss, curr_loss)

    # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
    if j % 2000 == 0:

        print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

        # The number of dinosaur names to print
        seed = 0
        for name in range(dino_names):
            # Sample indices and print them
            # Sample function iterate loop until: while (idx != newline_character and counter != 50):
            sampled_indices = sample(parameters, char_to_ix, seed)
            last_dino_name = ''.join(ix_to_char[ix] for ix in sampled_indices)
            last_dino_name = last_dino_name[0].upper() + last_dino_name[1:]  # capitalize first character
            print(last_dino_name.replace('\n', ''))

            seed += 1  # To get the same result (for grading purposes), increment the seed by one.

        print('\n')

