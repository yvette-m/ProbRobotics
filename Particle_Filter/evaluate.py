# Yvette Roos
# CS141 Probabilistic Robotics
# 3/10/26

import numpy as np
import cv2
import matplotlib.pyplot as plt
from particlefilter import (
    init_particles, random_position, compute_weights,
    check_convergence, resample, move_drone, move_particles,
    INJECT_COUNT, MAX_ITERATIONS
)

MAP_FILE   = 'CityMap.png'
TRIALS     = 20   # trials per condition — increase for smoother results


def run_trial(img, n_particles):
    """Run one headless trial, return iterations to converge (or MAX_ITERATIONS if failed)."""
    drone_pos = random_position(img)
    particles = init_particles(n_particles, img)

    for iteration in range(1, MAX_ITERATIONS + 1):
        weights = compute_weights(particles, drone_pos, img)

        if check_convergence(particles, drone_pos):
            return iteration, True

        particles = resample(particles, weights, img)
        drone_pos, unit_vec = move_drone(drone_pos, img)
        particles = move_particles(particles, unit_vec, img)

    return MAX_ITERATIONS, False  # did not converge


def experiment_particle_count(img):
    """Experiment 1: vary N particles, measure iterations to convergence."""
    conditions = [250, 500, 1000]
    results = {}

    for n in conditions:
        print(f"  N={n}: ", end='', flush=True)
        iters = []
        converged = 0
        for t in range(TRIALS):
            it, success = run_trial(img, n)
            iters.append(it)
            if success:
                converged += 1
            print('.', end='', flush=True)
        results[n] = iters
        print(f"  {converged}/{TRIALS} converged, mean={np.mean(iters):.1f} iters")

    return conditions, results


def experiment_patch_size(img):
    """Experiment 2: vary vision radius (patch size), measure iterations to convergence."""
    import particlefilter as pf
    conditions = [20, 35, 50]
    results = {}

    original_radius = pf.DRONE_VISION_RADIUS

    for r in conditions:
        pf.DRONE_VISION_RADIUS = r
        print(f"  radius={r}px: ", end='', flush=True)
        iters = []
        converged = 0
        for t in range(TRIALS):
            it, success = run_trial(img, 1000)
            iters.append(it)
            if success:
                converged += 1
            print('.', end='', flush=True)
        results[r] = iters
        print(f"  {converged}/{TRIALS} converged, mean={np.mean(iters):.1f} iters")

    pf.DRONE_VISION_RADIUS = original_radius  # restore
    return conditions, results


def plot_results(conditions_1, results_1, conditions_2, results_2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Experiment 1 — particle count
    ax = axes[0]
    means  = [np.mean(results_1[n]) for n in conditions_1]
    bars = ax.bar([str(n) for n in conditions_1], means,
                  color=['#6a5acd', '#836dcd', '#9d85cd'])
    ax.set_xlabel('Number of particles (N)')
    ax.set_ylabel('Iterations to convergence')
    ax.set_title('Experiment 1: Particle Count vs Convergence Speed')

    # Experiment 2 — patch size
    ax = axes[1]
    means2 = [np.mean(results_2[r]) for r in conditions_2]
    bars2 = ax.bar([f'{r}px\n({r*2}×{r*2})' for r in conditions_2], means2,
                   color=['#cd5a5a', '#cd7a6d', '#cd9a85'])
    ax.set_xlabel('Vision radius (patch size m×m)')
    ax.set_ylabel('Iterations to convergence')
    ax.set_title('Experiment 2: Patch Size vs Convergence Speed')

    plt.tight_layout()
    plt.savefig('eval_results.png', dpi=150)
    print("\nSaved eval_results.png")
    plt.show()


def main():
    img = cv2.imread(MAP_FILE)
    if img is None:
        raise FileNotFoundError(f"Could not load {MAP_FILE}")

    print(f"Running {TRIALS} trials per condition on {MAP_FILE}")
    print(f"Metric: mean iterations to convergence (lower = better)\n")

    print("Experiment 1: varying particle count")
    cond1, res1 = experiment_particle_count(img)

    print("\nExperiment 2: varying patch size (vision radius)")
    cond2, res2 = experiment_patch_size(img)

    plot_results(cond1, res1, cond2, res2)


if __name__ == '__main__':
    main()
