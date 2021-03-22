import numpy as np
import matplotlib.pyplot as plt


def paint(grid, title="", vmin=None, vmax=None):
    plt.imshow(grid.T, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show()


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


def collision(grid, eps=1e-7):
    density = grid.sum(axis=0)
    u = directions.T.dot(np.swapaxes(grid, 0, 1)) / (density+eps)
    eq_pop = equilibrium(density, u)
    new_pops = grid + omega * (eq_pop - grid)

    return new_pops


def inflow_left(grid, speed, obstacle):
    left_dirs = [0, 3, 6]
    middle_dirs = [1, 4, 7]
    right_dirs = [2, 5, 8]

    density = (grid[middle_dirs, 0].sum(axis=0) + 2 * grid[left_dirs, 0].sum(axis=0)) / (1 - speed)
    u = np.zeros((2, 1, grid.shape[2]))
    u[0] = speed
    eq_pop = equilibrium(density, u)

    not_wall = 1 - obstacle[0]
    for j in right_dirs:
        grid[j, 0][not_wall] = (eq_pop[j] + (grid[8-j, 0] - eq_pop[8-j]))[:, not_wall]

    return grid


directions = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
eq_prob = np.array([1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

Nx, Ny = 10, 10
omega = 0.4
Ux, Uy = 0.4, 0

initial_velocities = np.zeros((2, Nx, Ny))
initial_velocities[0], initial_velocities[1] = Ux, Uy
v_pops = equilibrium(1, initial_velocities)
# v_pops[:, Nx//2-5:Nx//2+5, Ny//2-5:Ny//2+5] = 0.4

wall = np.zeros((Nx, Ny), dtype=int)
# wall = np.array([[(x-Nx//2)**2+(y-Ny//2)**2 < 10**2 for y in range(Ny)] for x in range(Nx)], dtype=int)
wall[-1, :], wall[0, :], wall[:, 0], wall[:, -1] = 0, 0, 1, 1  # Left, Right, Bottom, Top
for i, slice in enumerate(v_pops):
    v_pops[i][wall.nonzero()] = 0

paint(wall)

vmin, vmax = np.min(v_pops.sum(axis=0)) - 1 / Nx / Ny, np.max(v_pops.sum(axis=0))*2 - 1 / Nx / Ny
paint(v_pops.sum(axis=0) - 1 / Nx / Ny, vmin=vmin, vmax=vmax)

for i in range(100):
    print(str(i) + " " + str((v_pops > 1e4).any()))
    collision_pops = collision(v_pops)
    v_pops = inflow_left(v_pops, Ux, wall)
    v_pops = obstacles(v_pops, collision_pops, wall)
    v_pops = stream(v_pops)

    if i % 10 == 0:
        paint(np.sum(v_pops, axis=0) - 1 / Nx / Ny, title=str(i), vmin=vmin, vmax=vmax)
