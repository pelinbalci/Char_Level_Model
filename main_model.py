import numpy as np
import random
import pprint
from sample_output import sample
from helper_functions import initialize_parameters, get_sample, get_initial_loss, smooth
from rnn_helper import optimize


def model(data_x, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27,
          verbose=False):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data_x -- text corpus, divided in words
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text (size of the vocabulary)

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)

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

        ### START CODE HERE ###

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
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        ### END CODE HERE ###

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
                last_dino_name = get_sample(sampled_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))

                seed += 1  # To get the same result (for grading purposes), increment the seed by one.

            print('\n')

    return parameters, last_dino_name


data = open('dinos.txt', 'r').read()
data= data.lower()
names = [x.strip() for x in data.split("\n")]
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)

parameters, last_name = model(data.split("\n"), ix_to_char, char_to_ix, 22001, verbose = True)

print('Done')



