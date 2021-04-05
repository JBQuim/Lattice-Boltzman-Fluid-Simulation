import numpy as np
import matplotlib.pyplot as plt
import LBFluidsTools as lb


Nx, Ny = 200, 80
wall = np.array([[(x - Nx // 5) ** 2 + (y - Ny // 2) ** 2 < (Ny // 10) ** 2 for y in range(Ny)] for x in range(Nx)],
                dtype=int)

max_it, interval, wait_it = 10000, 200, 2000

sim = lb.D2Q9(Nx, Ny, 1.95, 0.04, wall)
history_force_x = np.zeros(max_it)
history_force_y = np.zeros(max_it)
for i in range(max_it):
    print(i, (sim.v_pops < 0).any())
    sim.step()
    history_force_x[i], history_force_y[i] = sim.force[0], sim.force[1]

    if i % interval == 0 and i >= wait_it:
        coords = (sim.com_x, sim.com_y, history_force_x[i-interval:i].mean() * 1500, history_force_y[i-interval:i].mean() * 1500)
        # paint(velocities[0], vmin=-Ux * 4, vmax=Ux * 4, title=str(i))
        # print(coords)
        # lb.paint(sim.velocities[0], title=str(i), arrow=coords)
        # lb.paint(sim.get_curl(), title=str(i), vmin=-0.02, vmax=0.02, arrow=coords)
        # lb.paint(sim.get_density(), title=i)
        # lb.paint(sim.get_density(), title=str(i))

plt.plot(np.convolve(history_force_x, np.ones(10), 'valid'))
plt.show()
plt.plot(np.convolve(history_force_y, np.ones(10), 'valid'))
plt.show()
