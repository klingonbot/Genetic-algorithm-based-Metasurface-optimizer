import numpy as np


def compute_pec_peak(S_matrix, theta_deg, phi_deg):
   
    n_bits = S_matrix.shape[0]
    gamma_pec = np.ones(n_bits, dtype=complex)
    af_pec = gamma_pec @ S_matrix.reshape(n_bits, -1)
    af_pec_map = af_pec.reshape(len(theta_deg), len(phi_deg))
    pec_peak = np.max(np.abs(af_pec_map) ** 2)
    return pec_peak, af_pec_map


def individual_to_gamma(individual, B):
   
    phase_levels = 2 * np.pi * individual / (2 ** B)
    return np.exp(1j * phase_levels)


def evaluate_individual(individual, S_matrix, theta_deg, phi_deg, B, pec_peak):
    
    n_bits = len(individual)
    gamma = individual_to_gamma(individual, B)
    af = gamma @ S_matrix.reshape(n_bits, -1)
    af_map = af.reshape(len(theta_deg), len(phi_deg))
    candidate_peak = np.max(np.abs(af_map) ** 2)
    eps = 1e-12
    fitness = 10 * np.log10((pec_peak + eps) / (candidate_peak + eps))
    return fitness, af_map, candidate_peak


def evaluate_population(population, S_matrix, theta_deg, phi_deg, B, pec_peak):
    
    pop_size = population.shape[0]
    fitness_values = np.zeros(pop_size)
    af_maps = []
    candidate_peaks = np.zeros(pop_size)

    for i in range(pop_size):
        fitness_values[i], af_map, candidate_peaks[i] = evaluate_individual(
            population[i], S_matrix, theta_deg, phi_deg, B, pec_peak
        )
        af_maps.append(af_map)

    return fitness_values, af_maps, candidate_peaks
