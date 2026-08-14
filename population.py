import numpy as np


def initialize_population(pop_size, N, M, B):
   

    Seeds the population with four structured patterns (all-zero,
    checkerboard, alternating rows, alternating columns) and fills the
    remainder with random individuals.

    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    N, M : int
        Array dimensions.
    B : int
        Number of coding bits per element (2**B phase states).

    Returns
    -------
    population : ndarray, shape (pop_size, N*M)
        Integer-coded population, values in [0, 2**B).
    """
    n_bits = N * M
    n_states = 2 ** B

    population = []


    population.append(np.zeros(n_bits, dtype=int))


    checkerboard = np.fromfunction(
        lambda i, j: (i + j) % n_states,
        (N, M),
        dtype=int
    ).flatten()
    population.append(checkerboard)

 
    row_pattern = np.fromfunction(
        lambda i, j: i % n_states,
        (N, M),
        dtype=int
    ).flatten()
    population.append(row_pattern)

    col_pattern = np.fromfunction(
        lambda i, j: j % n_states,
        (N, M),
        dtype=int
    ).flatten()
    population.append(col_pattern)

 
    while len(population) < pop_size:
        population.append(
            np.random.randint(
                0,
                n_states,
                size=n_bits
            )
        )

    return np.array(population[:pop_size])
