"""
This is a python implementation of the poisson-disc algorithm.
Poisson-disc code is borrowed from
IHautal at https://github.com/IHautaI/poisson-disc

Poisson-disc algorithm produces points in a grid,
but no closer to each other than a specified minimum distance

For more details about this algorithm :
http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
"""

import random
from math import sqrt, pi, sin, cos
from itertools import product


def generate(grid):
    """
    build the grid for generating the points
    """
    def func(point):
        new = [random.choice([random.uniform(-grid.r*2, 0),
               random.uniform(0, grid.r*2)]) for _ in range(len(point))]
        return tuple(new[i] + point[i] for i in range(len(point)))
    return func


def generate_points(r, length, width, n_points, rand=None):
    """
    generate n_points over a grid of a given length and width
    """
    grid = Grid(r, length, width)
    grid.generate = generate(grid)
    if rand is None:
        rand = (random.uniform(0, length), random.uniform(0, width))

    for i in range(100):
        data = grid.poisson(rand, n_points)
        if len(data) != n_points:
            continue
        else:
            return data


class Grid:
    """
    class for filling a rectangular prism of dimension >= 2
    with poisson disc samples spaced at least r apart
    and k attempts per active sample
    override Grid.distance to change
    distance metric used and get different forms
    of 'discs'
    """
    def __init__(self, r, *size):
        self.r = r

        self.size = size
        self.dim = len(size)

        self.cell_size = r/(sqrt(self.dim))

        self.widths = [int(size[k]/self.cell_size)+1 for k in range(self.dim)]

        nums = product(*(range(self.widths[k]) for k in range(self.dim)))

        self.cells = {num: -1 for num in nums}
        self.samples = []
        self.active = []

    def clear(self):
        """
        resets the grid
        active points and
        sample points
        """
        self.samples = []
        self.active = []

        for item in self.cells:
            self.cells[item] = -1

    def generate(self, point):
        """
        generates new points
        in an annulus between
        self.r, 2*self.r
        """

        rad = random.triangular(self.r, 2*self.r, .3*(2*self.r - self.r))
        # was random.uniform(self.r, 2*self.r) but I think
        # this may be closer to the correct distribution
        # but easier to build

        angs = [random.uniform(0, 2*pi)]

        if self.dim > 2:
            angs.extend(random.uniform(-pi/2, pi/2) for _ in range(self.dim-2))

        angs[0] = 2*angs[0]

        return self.convert(point, rad, angs)

    def poisson(self, seed, k=30):
        """
        generates a set of poisson disc samples
        """
        self.clear()

        self.samples.append(seed)
        self.active.append(0)
        self.update(seed, 0)

        while len(self.samples) < k and self.active:

            idx = random.choice(self.active)
            point = self.samples[idx]
            new_point = self.make_points(k, point)

            if new_point:
                self.samples.append(tuple(new_point))
                self.active.append(len(self.samples)-1)
                self.update(new_point, len(self.samples)-1)
            else:
                self.active.remove(idx)

        return self.samples

    def make_points(self, k, point):
        """
        uses generate to make up to
        k new points, stopping
        when it finds a good sample
        using self.check
        """
        n = k

        while n:
            new_point = self.generate(point)
            if self.check(point, new_point):
                return new_point

            n -= 1

        return False

    def check(self, point, new_point):
        """
        checks the neighbors of the point
        and the new_point
        against the new_point
        returns True if none are closer than r
        """
        for i in range(self.dim):
            if not (0 < new_point[i] < self.size[i] or
               self.cellify(new_point) == -1):
                return False

        for item in self.neighbors(self.cellify(point)):
            if self.distance(self.samples[item], new_point) < self.r**2:
                return False

        for item in self.neighbors(self.cellify(new_point)):
            if self.distance(self.samples[item], new_point) < self.r**2:
                return False

        return True

    def convert(self, point, rad, angs):
        """
        converts the random point
        to rectangular coordinates
        from radial coordinates centered
        on the active point
        """
        new_point = [point[0] + rad*cos(angs[0]), point[1] + rad*sin(angs[0])]
        if len(angs) > 1:
            new_point.extend(point[i+1] + rad*sin(angs[i])
                             for i in range(1, len(angs)))
        return new_point

    def cellify(self, point):
        """
        returns the cell in which the point falls
        """
        return tuple(point[i]//self.cell_size for i in range(self.dim))

    def distance(self, tup1, tup2):
        """
        returns squared distance between two points
        """
        return sum((tup1[k] - tup2[k])**2 for k in range(self.dim))

    def cell_distance(self, tup1, tup2):
        """
        returns true if the L1 distance is less than 2
        for the two tuples
        """
        return sum(abs(tup1[k]-tup2[k]) for k in range(self.dim)) <= 2

    def neighbors(self, cell):
        """
        finds all occupied cells within
        a distance of the given point
        """
        return (self.cells[tup] for tup in self.cells
                if self.cells[tup] != -1 and
                self.cell_distance(cell, tup))

    def update(self, point, index):
        """
        updates the grid with the new point
        """
        self.cells[self.cellify(point)] = index

    def __str__(self):
        return self.cells.__str__()
