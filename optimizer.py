import numpy as np

from .steering import compute_steering_matrix
from .population import initialize_population
from .fitness import compute_pec_peak, evaluate_population, individual_to_gamma
from .genetic_operators import adaptive_mutation_rate, create_new_population


def optimize_rcs(N=10, M=10, B=2, pop_size=200, max_generations=100,
                  dx=0.5, dy=0.5,
                  theta_range=(0, 90, 19),
                  phi_range=(0, 360, 37),
                  crossover_rate=0.9,
                  mutation_rate=0.01,
                  tournament_size=3):
    """
    Run the genetic algorithm to find a coding-metasurface phase pattern
    that minimizes the array-factor peak (i.e., maximizes RCS reduction in
    dB relative to a PEC reference).

    Returns
    -------
    dict with keys:
        S_matrix, theta_deg, phi_deg, PEC_peak, PEC_map,
        best_fitness_dB, best_generation, best_individual,
        best_AF_map, best_candidate_peak, optimized_matrix,
        rcs_reduction_dB, N, M
    """
    S_matrix, theta_deg, phi_deg = compute_steering_matrix(
        N=N, M=M, dx=dx, dy=dy,
        theta_range=theta_range, phi_range=phi_range
    )

    pec_peak, pec_map = compute_pec_peak(S_matrix, theta_deg, phi_deg)

    population = initialize_population(pop_size, N, M, B)

    best_fitness = -np.inf
    best_individual = None
    best_generation = -1
    best_af_map = None
    best_candidate_peak = None

    for generation in range(max_generations):
        fitness_values, af_maps, candidate_peaks = evaluate_population(
            population, S_matrix, theta_deg, phi_deg, B, pec_peak
        )

        current_gen_best_idx = np.argmax(fitness_values)
        current_gen_best_fitness = fitness_values[current_gen_best_idx]

        if current_gen_best_fitness > best_fitness:
            best_fitness = current_gen_best_fitness
            best_individual = population[current_gen_best_idx].copy()
            best_generation = generation
            best_af_map = af_maps[current_gen_best_idx]
            best_candidate_peak = candidate_peaks[current_gen_best_idx]

        if generation == max_generations - 1:
            break

        current_mutation_rate = adaptive_mutation_rate(
            generation,
            max_generations,
            start_rate=0.25,
            end_rate=0.01
        )

        population = create_new_population(
            population,
            fitness_values,
            B,
            elitism=True,
            crossover_rate=crossover_rate,
            mutation_rate=current_mutation_rate,
            tournament_size=tournament_size,
            pec_peak=pec_peak,
            S_matrix=S_matrix,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            elite_protection_threshold=0.95
        )

    optimized_matrix = individual_to_gamma(best_individual, B)
    rcs_reduction_db = best_fitness

    return {
        'S_matrix': S_matrix,
        'theta_deg': theta_deg,
        'phi_deg': phi_deg,
        'PEC_peak': pec_peak,
        'PEC_map': pec_map,
        'best_fitness_dB': best_fitness,
        'best_generation': best_generation,
        'best_individual': best_individual,
        'best_AF_map': best_af_map,
        'best_candidate_peak': best_candidate_peak,
        'optimized_matrix': optimized_matrix,
        'rcs_reduction_dB': rcs_reduction_db,
        'N': N,
        'M': M
    }
