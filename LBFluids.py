import numpy as np
import matplotlib.pyplot as plt
import LBFluidsTools as lb


Nx, Ny = 200, 80
wall = np.array([[(x - Nx // 5) ** 2 + (y - Ny // 2) ** 2 < (Ny // 10) ** 2 for y in range(Ny)] for x in range(Nx)],
                dtype=int)
wall[-1, :], wall[0, :] = 0, 0  # Left, Right
wall[:, 0], wall[:, -1] = 0, 0  # Bottom, Top

max_it, interval, wait_it = 20000, 100, 2000

sim = lb.D2Q9(Nx, Ny, 1.95, 0.04, wall)
for i in range(max_it):
    print(i)
    sim.step()

    if i % interval == 0 and i >= wait_it:
        # paint(velocities[0], vmin=-Ux * 4, vmax=Ux * 4, title=str(i))
        # paint(velocities[1], vmin=-Ux * 2, vmax=Ux * 2, title=str(i))
        lb.paint(sim.get_curl(), title=str(i), vmin=-0.02, vmax=0.02)
        # paint(density, title=i)
