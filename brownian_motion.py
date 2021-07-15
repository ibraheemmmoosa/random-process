import jax
import jax.numpy as jnp
from jax.random import uniform, randint, multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def simulate_brownian_motion(key, num_particles, num_steps, max_spread_start_position):
    mean = jnp.tile(jnp.array([0.0, 0.0]), (num_particles, 1))
    cov = jnp.tile(jnp.array([[1.0, 0.0], [0.0, 1.0]]), (num_particles, 1, 1))

    random_steps = multivariate_normal(key, mean, cov, shape=(num_steps, num_particles))
    cumulative_steps = jnp.cumsum(random_steps, axis=0) 

    start_positions = uniform(key, shape=(num_particles, 2), minval=-max_spread_start_position, maxval=max_spread_start_position)
    positions = start_positions + cumulative_steps
    
    return positions

def visualize_brownian_motion(key, positions, wall_position, animation_interval, particle_colors=None):
    num_steps, num_particles, _ = positions.shape
    start_positions = positions[0, :, :]
    if particle_colors == None:
        particle_colors = randint(key, shape=(num_particles,), minval=0, maxval=100, dtype=jnp.int32)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(xlim=(-wall_position, wall_position), ylim=(-wall_position, wall_position))
    scatter_plot = ax.scatter(start_positions[:, 0], start_positions[:, 1], s=particle_size, c=particle_colors, cmap='viridis')

    def update_scatter_plot(frame_number):
        scatter_plot.set_offsets(positions[frame_number, :, :])

    anim = FuncAnimation(fig, update_scatter_plot, interval=animation_interval)
    plt.show()


if __name__ == '__main__':
    num_particles = 100
    particle_size = 100
    num_steps = 10000
    max_spread_start_position = 10.0
    wall_position = 100.0
    animation_interval = 10

    key = jax.random.PRNGKey(0)

    positions = simulate_brownian_motion(key, num_particles, num_steps, max_spread_start_position)
    particle_colors = positions[0, :, 0]
    visualize_brownian_motion(key, positions, wall_position, animation_interval, particle_colors)
