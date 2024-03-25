import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from utils import compute_distance
import os

def visualize_pareto_front(candidate_solutions, optimal_solutions, iteration):
    """
    Visualize the pareto front of the candidate solutions in a scatter plot.
    X-axis: distance
    Y-axis: fitness
    The optimal solutions are highlighted in red.
    """
    plt.scatter([candidate_solutions[key][0] for key in candidate_solutions.keys()], [candidate_solutions[key][1] for key in candidate_solutions.keys()], label="Candidate solutions")
    plt.scatter([candidate_solutions[key][0] for key in optimal_solutions], [candidate_solutions[key][1] for key in optimal_solutions], color="red", label="Optimal solutions")
    plt.xlabel("Distance")
    plt.ylabel("Fitness")
    plt.title(f"Pareto front of candidate solutions itr={iteration}")
    plt.legend()
    plt.show()

def visualize_population(solutions):
    # Visualize the entire population using MDS
    distance_map = np.zeros((len(solutions), len(solutions)))

    for i, key1 in enumerate(solutions.keys()):
        for j, key2 in enumerate(solutions.keys()):
            distance_map[i][j] = compute_distance(solutions[key1][0], solutions[key2][0])

    mds = MDS(n_components=2, dissimilarity="precomputed")
    embedding = mds.fit_transform(distance_map)
    # Get color based on the fitness
    fitness = [solutions[key][1] for key in solutions.keys()]
    v_min = min(fitness); v_max = max(fitness)

    # Plot the MDS embedding
    plt.scatter(embedding[:, 0], embedding[:, 1], c=fitness, vmin=v_min, vmax=v_max, cmap="viridis")
    plt.xlabel("mds 1")
    plt.ylabel("mds 2")
    plt.title("Parameter space embedding of all solutions")
    plt.show()


def visualize_ensemble(solutions, ensemble_set):
    # Visualize ensemble using MDS
    distance_map = np.zeros((len(ensemble_set), len(ensemble_set)))
    for i, key1 in enumerate(ensemble_set):
        for j, key2 in enumerate(ensemble_set):
            distance_map[i][j] = compute_distance(solutions[key1][0], solutions[key2][0])

    #plt.imshow(distance_map, cmap="hot", interpolation="nearest")
    #plt.show()

    mds = MDS(n_components=2, dissimilarity="precomputed")
    embedding = mds.fit_transform(distance_map)

    fitness = [solutions[key][1] for key in ensemble_set]
    all_fitness = [solutions[key][1] for key in solutions.keys()]
    v_min, v_max = min(all_fitness), max(all_fitness)

    # Plot the MDS embedding
    plt.scatter(embedding[:, 0], embedding[:, 1], c=fitness, vmin=v_min, vmax=v_max, cmap="viridis")
    plt.xlabel("mds 1")
    plt.ylabel("mds 2")
    plt.title("Parameter space embedding of ensemble solutions")
    plt.show()

def combine_plots(solutions, ensemble, candidate_solutions, optimal_solutions, included, iteration, dst_path):
    """
    Combines the ensemble plot, population plot, and pareto front plot into a single figure per iteration
    Will be used to create a gif of the optimization process
    All of the frames will be saved in the dst_path
    """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    all_fitness = [solutions[key][1] for key in solutions.keys()]
    v_min, v_max = min(all_fitness), max(all_fitness)

    # Get the ensemble embedding
    distance_map = np.zeros((len(ensemble), len(ensemble)))
    for i, key1 in enumerate(ensemble):
        for j, key2 in enumerate(ensemble):
            distance_map[i][j] = compute_distance(solutions[key1][0], solutions[key2][0])

    mds = MDS(n_components=2, dissimilarity="precomputed")
    ens_embedding = mds.fit_transform(distance_map)
    ens_fitness = [solutions[key][1] for key in ensemble]

    # Get the population embedding
    distance_map = np.zeros((len(solutions), len(solutions)))
    for i, key1 in enumerate(solutions.keys()):
        for j, key2 in enumerate(solutions.keys()):
            distance_map[i][j] = compute_distance(solutions[key1][0], solutions[key2][0])

    mds = MDS(n_components=2, dissimilarity="precomputed")
    pop_embedding = mds.fit_transform(distance_map)

    # Get the pareto front
    candidate_solutions_x = [candidate_solutions[key][0] for key in candidate_solutions.keys()]
    candidate_solutions_y = [candidate_solutions[key][1] for key in candidate_solutions.keys()]

    optimal_solutions_x = [candidate_solutions[key][0] for key in optimal_solutions]
    optimal_solutions_y = [candidate_solutions[key][1] for key in optimal_solutions]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Add more spacing between the plots
    #fig.subplots_adjust(wspace=0.5)

    # Title for the entire figure
    fig.suptitle(f"Ensemble selection itr={iteration}")

    # Plot the population embedding
    ax[0].scatter(pop_embedding[:, 0], pop_embedding[:, 1], c=all_fitness, vmin=v_min, vmax=v_max, cmap="viridis")
    ax[0].set_xlabel("mds 1")
    ax[0].set_ylabel("mds 2")
    ax[0].set_title("Parameter space embedding of candidate solutions")

    # Plot the ensemble embedding
    ax[1].scatter(ens_embedding[:, 0], ens_embedding[:, 1], c=ens_fitness, vmin=v_min, vmax=v_max, cmap="viridis")
    ax[1].set_xlabel("mds 1")
    ax[1].set_ylabel("mds 2")
    ax[1].set_title("Parameter space embedding of ensemble")


    # Plot the pareto front
    ax[2].scatter(candidate_solutions_x, candidate_solutions_y, label="Candidate solutions")
    ax[2].scatter(optimal_solutions_x, optimal_solutions_y, color="red", label="Optimal solutions")
    ax[2].set_xlabel("Distance")
    ax[2].set_ylabel("Fitness")
    ax[2].set_title(f"Pareto front of candidate solutions itr={iteration}")
    ax[2].legend()

    plt.savefig(f"{dst_path}/iteration_{iteration}.png")
