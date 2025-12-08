import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

n_steps = 1000
# Probability that a particle births another particle at each step
p_birth = 0.0001
# Probability that a particle dies at each step
p_death = 0.0001

for run in range(100):
    # Run simulation
    n_particles = 100
    alive_population = [n_particles]
    dead_population = [0]
    for step in range(n_steps):
        # Birth new particles
        # Death of particles
        births = np.random.rand(n_particles) < p_birth
        deaths = np.random.rand(n_particles) < p_death
        n_particles += np.sum(births)
        n_particles -= np.sum(deaths)

        # Ensure non-negative population
        n_particles = max(n_particles, 0)

        # Store population data for plotting
        alive_population.append(n_particles)
        dead_population.append(dead_population[-1] + np.sum(deaths))
        
    # Plot alive and dead population over time
    plt.plot(alive_population, color='green')
    # plt.plot(dead_population, label='Dead Population', color='red')


plt.xlabel('Time Steps')
plt.ylabel('Number of Particles')
plt.title('Particle Birth and Death Simulation')
plt.legend()
plt.grid()
plt.show()