import numpy as np
import matplotlib.pyplot as plt


def paint(grid, title="", vmin=None, vmax=None, arrow=None):
    plt.imshow(grid.T, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    if not (arrow is None):
        x, y, dx, dy = arrow
        plt.arrow(x, y, dx, dy, length_includes_head=True, color="w", overhang=1, head_width=3)
    plt.show()


def curl(velocities, obstacle=None):
    """
    Calculates curl of a velocity field.
    velocities: velocity field of size (2, X, Y)
    obstacle: mask of size (X, Y) where there is an obstacle. All adjacent points have 0 curl.
    Returns an array of shape (X, Y) with the curl at each point.
    """
    if obstacle is None:
        obstacle = np.zeros(velocities.shape[1:])

    # v = P(x,y)i + Q(x,y)j
    # curl = dQ/dx - dP/dy
    dQdx = (velocities[1, 2:] - velocities[1, :-2]) / 2
    dPdy = (velocities[0, :, 2:] - velocities[0, :, :-2]) / 2
    curl = dQdx[:, 1:-1] - dPdy[1:-1]

    # curl at edges is 0 as there is not enough information
    curl = np.pad(curl, 1)

    # curl next to obstacles is also 0
    curl = curl * (1 - obstacle)
    curl[1:] = curl[1:] * (1 - obstacle[:-1])
    curl[:-1] = curl[:-1] * (1 - obstacle[1:])
    curl[:, 1:] = curl[:, 1:] * (1 - obstacle[:, :-1])
    curl[:, :-1] = curl[:, :-1] * (1 - obstacle[:, 1:])

    return curl


class D2Q9:
    def __init__(self, Nx, Ny, omega, Ux, wall=None, w=1e-3):
        self.left_dirs = [0, 3, 6]
        self.middle_dirs = [1, 4, 7]
        self.right_dirs = [2, 5, 8]
        self.directions = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
        self.eq_prob = np.array([1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

        self.Nx, self.Ny = Nx, Ny
        self.omega = omega
        self.Ux = Ux

        if wall is None:
            self.wall = np.zeros((Nx, Ny), dtype=int)
        else:
            self.wall = wall

        self.force = np.array([0, 0])
        # outside the walls the velocity is set to equilibrium with horizontal speed Ux, inside the walls at rest
        # this makes the calculation of the force on the walls convenient
        self.density = np.ones((Nx, Ny))
        self.velocities = np.zeros((2, Nx, Ny))
        self.velocities[0] = Ux * (1 - self.wall)
        self.v_pops = self.equilibrium(self.density, self.velocities) + w * np.random.rand(9, Nx, Ny)

        self.yes_mask = wall.nonzero()
        self.no_mask = (1 - wall).nonzero()
        self.com_x, self.com_y = np.mean(self.yes_mask, axis=1)

    def get_density(self):
        return self.v_pops.sum(axis=0)

    def get_velocity(self, eps=1e-5, density=None):
        if density is None:
            density = self.get_density()
        return self.directions.T.dot(np.swapaxes(self.v_pops, 0, 1)) / (density + eps)

    def get_curl(self):
        return curl(self.velocities, obstacle=self.wall)

    def equilibrium(self, density, u):
        u_squared = np.sum(u ** 2, axis=0)
        dir_dot_u = self.directions.dot(np.swapaxes(u, 0, 1))

        return density * self.eq_prob[:, np.newaxis, np.newaxis] * \
               (1 + 3 * dir_dot_u + 9 / 2 * dir_dot_u ** 2 - 3 / 2 * u_squared)

    def inflow_left(self, eq_slice):
        not_wall = (1 - self.wall[0]).nonzero()
        v_slice = self.v_pops[:, 0]
        for j in self.right_dirs:
            deviation = v_slice[8 - j] - eq_slice[8 - j]
            v_slice[j, not_wall] = (eq_slice[j] + deviation)[not_wall]

        return v_slice

    def stream(self):
        grid = self.v_pops
        rolled_grid = np.zeros_like(grid)

        for j, slice_2d in enumerate(grid):
            rolled_grid[j] = np.roll(slice_2d, self.directions[j], axis=(0, 1))

        return rolled_grid

    def obstacles(self, post_collision):
        in_wall = self.yes_mask
        for j in range(9):
            post_collision[j][in_wall] = self.v_pops[8-j][in_wall]

        return post_collision

    def step(self):
        density = self.get_density()
        self.velocities = self.get_velocity(density=density)

        # Inflow condition
        density[0] = (self.v_pops[self.middle_dirs, 0].sum(axis=0) + 2 * self.v_pops[self.left_dirs, 0].sum(axis=0)) / (
                    1 - self.Ux)
        self.velocities[0, 0] = self.Ux

        eq_pop = self.equilibrium(density, self.velocities)
        self.v_pops[:, 0] = self.inflow_left(eq_pop[:, 0])

        # Collision
        new_pops = self.v_pops - self.omega * (self.v_pops - eq_pop)

        # Obstacle
        self.v_pops = self.obstacles(new_pops)

        # Calculate force on object
        self.force = (self.velocities * self.wall).sum(axis=(1,2))

        # Streaming
        self.v_pops = self.stream()

        # outflow at the right
        self.v_pops[self.left_dirs, -1] = self.v_pops[self.left_dirs, -2]


def naca_4(a, b, c, N=10000):
    """
    Generates N points along an airfoil, according to NACA 4 digit specifications.
    a: First digit
    b: Second digit
    c: Third and fourth digits.
    Output: 4 arrays, xu and yu hold the upper surface coordintaes. xl and yl the lower surface. The x coordinate is
    scaled so the whole airfoil is between 0 and 1.
    """

    m = a / 100
    p = b / 10
    t = c / 100
    beta = np.pi * np.arange(N + 1) / N
    x = 1 / 2 * (1 - np.cos(beta))

    y_t = 5 * t * (0.2969 * x ** (1 / 2) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)

    mask = x > p

    y_c = m / p ** 2 * (2 * p * x - x ** 2)
    y_c[mask] = (m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))[mask]

    dy_cdx = 2 * m / p ** 2 * (p - x)
    dy_cdx[mask] = (2 * m / (1 - p) ** 2 * (p - x))[mask]

    theta = np.arctan(dy_cdx)
    xu = x - y_t * np.sin(theta)
    yu = y_c + y_t * np.cos(theta)
    xl = x + y_t * np.sin(theta)
    yl = y_c - y_t * np.cos(theta)

    # recentering and rescaling
    xmin = xu.min()
    xu -= xmin
    xl -= xmin
    xmax = xu.max()
    yu -= yl.min()
    yl -= yl.min()

    return xu / xmax, yu / xmax, xl / xmax, yl / xmax


def image_naca_4(a, b, c, Nx):
    """
    Generates a boolean image of an airfoil, according to NACA specifications. Image is Nx wide.
    """
    spacing = 1 / (Nx - 1)
    x = np.arange(Nx) * spacing

    xu, yu, xl, yl = naca_4(a, b, c)
    yu = np.interp(x, xu, yu)
    yl = np.interp(x, xl, yl)

    Ny = int(yu.max() / spacing) + 1
    grid = np.repeat(np.arange(Ny)[:, np.newaxis], Nx, axis=1) * spacing
    return (grid >= yl) * (grid <= yu)