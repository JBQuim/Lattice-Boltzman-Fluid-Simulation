import numpy as np
import matplotlib.pyplot as plt


def paint(grid, title="", vmin=None, vmax=None):
    plt.imshow(grid.T, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show()


def get_density(grid):
    return grid.sum(axis=0)


def get_velocity(grid, eps=1e-6):
    return directions.T.dot(np.swapaxes(grid, 0, 1)) / (density + eps)


def stream(grid):
    rolled_grid = np.zeros_like(grid)
    for j, slice_2d in enumerate(grid):
        rolled_grid[j] = np.roll(slice_2d, directions[j], axis=(0, 1))

    return rolled_grid


def obstacles(pre_collision, post_collision, obstacle):
    in_wall = obstacle.nonzero()
    not_wall = (1 - obstacle).nonzero()

    new_grid = np.zeros_like(pre_collision)
    for j in range(9):
        new_grid[8 - j][in_wall] = pre_collision[j][in_wall]
        new_grid[j][not_wall] = post_collision[j][not_wall]

    return new_grid


def equilibrium(density, u):
    u_squared = np.sum(u ** 2, axis=0)
    dir_dot_u = directions.dot(np.swapaxes(u, 0, 1))

    return density * eq_prob[:, np.newaxis, np.newaxis] * \
           (1 + 3 * dir_dot_u + 9 / 2 * dir_dot_u ** 2 - 3 / 2 * u_squared)


def inflow_left(v_slice, eq_slice, obstacle):
    not_wall = (1 - obstacle).nonzero()
    for j in right_dirs:
        deviation = v_slice[8 - j] - eq_slice[8 - j]
        v_slice[j, not_wall] = (eq_slice[j] + deviation)[not_wall]

    return v_slice


directions = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
eq_prob = np.array([1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36])
left_dirs = [0, 3, 6]
middle_dirs = [1, 4, 7]
right_dirs = [2, 5, 8]

Nx, Ny = 200, 40
omega = 1.95
Ux = 0.04

initial_velocities = np.zeros((2, Nx, Ny))
initial_velocities[0] = Ux/100

v_pops = equilibrium(1, initial_velocities) + 1e-1 * np.random.rand(9, Nx, Ny)
# v_pops[:, Nx // 2 - 5:Nx // 2 + 5, Ny // 2 - 5:Ny // 2 + 5] = 0.4?

wall = np.zeros((Nx, Ny), dtype=int)
wall = np.array([[(x-Nx//5)**2+(y-Ny//2)**2 < (Ny//5)**2 for y in range(Ny)] for x in range(Nx)], dtype=int)
wall[-1, :], wall[0, :] = 0, 0  # Left, Right
wall[:, 0], wall[:, -1] = 1, 1  # Bottom, Top

for i, slice in enumerate(v_pops):
    v_pops[i][wall.nonzero()] = 0
paint(wall)

vmin, vmax = np.min(v_pops.sum(axis=0)) - 1 / Nx / Ny, np.max(v_pops.sum(axis=0)) - 1 / Nx / Ny
paint(v_pops.sum(axis=0) - 1 / Nx / Ny, vmin=vmin, vmax=vmax)

for i in range(10000):
    print(str(i) + " " + str((v_pops > 1e4).any()))

    density = get_density(v_pops)
    velocities = get_velocity(v_pops)

    # Inflow condition
    density[0] = (v_pops[middle_dirs, 0].sum(axis=0) + 2 * v_pops[left_dirs, 0].sum(axis=0)) / (1 - Ux)
    velocities[0, 0] = Ux

    eq_pop = equilibrium(density, velocities)
    v_pops[:, 0] = inflow_left(v_pops[:, 0], eq_pop[:, 0], wall[0])

    # collision
    new_pops = v_pops - omega * (v_pops - eq_pop)

    # obstacles
    v_pops = obstacles(v_pops, new_pops, wall)

    # streaming
    v_pops = stream(v_pops)

    # outflow at the right
    v_pops[left_dirs, -1] = v_pops[left_dirs, -2]

    if i % 100 == 0 and i > 2000:
        # paint(velocities[0], vmin=-Ux * 4, vmax=Ux * 4, title=str(i))
        paint(velocities[1], vmin=-Ux * 2, vmax=Ux * 2, title=str(i))
