#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tqdm


def fitness(length):
    return 1 / length


def route_length(route, distance_matrix):
    n = route.size
    idx = np.concatenate((route, [route[0]]))
    length = np.sum(distance_matrix[idx[:n], idx[1:n+1]])
    # length = 0
    # for i in range(n - 1):
        # length += distance_matrix[route[i], route[i+1]]
    # length += distance_matrix[route[n-1], route[0]]
    return length


def pmx_cross(routes):
    n = routes[0].size
    i, j = np.random.choice(n, 2)
    if i > j:
        i, j = j, i
    j += 1
    new_routes = routes.copy()

    new_routes[0, i:j] = routes[1, i:j]
    new_routes[1, i:j] = routes[0, i:j]

    mapping = np.array(range(n), dtype=int)
    mapping[new_routes[0, i:j]] = routes[0, i:j]
    for k in range(i):
        while new_routes[0, k] != mapping[new_routes[0, k]]:
            new_routes[0, k] = mapping[new_routes[0, k]]
    for k in range(j, n):
        while new_routes[0, k] != mapping[new_routes[0, k]]:
            new_routes[0, k] = mapping[new_routes[0, k]]

    mapping = np.array(range(n), dtype=int)
    mapping[new_routes[1, i:j]] = routes[1, i:j]
    for k in range(i):
        while new_routes[1, k] != mapping[new_routes[1, k]]:
            new_routes[1, k] = mapping[new_routes[1, k]]
    for k in range(j, n):
        while new_routes[1, k] != mapping[new_routes[1, k]]:
            new_routes[1, k] = mapping[new_routes[1, k]]

    return new_routes


def cx_cross(routes):
    def _cx_impl(routes):
        n = routes[0].size
        new_routes = np.zeros(n, dtype=int) - 1
        mapped = np.zeros(n, dtype=bool)
        i = np.random.randint(n)
        while not mapped[routes[0][i]]:
            new_routes[i] = routes[0][i]
            mapped[routes[0][i]] = True
            i = routes[1][i]
        j = 0
        for i in range(n):
            if new_routes[i] == -1:
                while mapped[routes[1][j]]:
                    j += 1
                new_routes[i] = routes[1][j]
                mapped[routes[1][j]] = True
        return new_routes
    return [_cx_impl(routes), _cx_impl(np.flip(routes))]


def cross(routes):
    return pmx_cross(routes)


def mutate(route):
    n = route.size
    i, j = np.random.choice(n, 2, replace=False)
    if i > j:
        i, j = j, i
    # route[i], route[j] = route[j], route[i]
    route[i:j+1] = np.flip(route[i:j+1])


def ga(distance_matrix, population_size, cross_rate, mutation_rate, max_iters, thres=-1):
    n = distance_matrix.shape[0]
    ncross = np.ceil(population_size * cross_rate * 0.5).astype(int) * 2
    nreserve = population_size - ncross
    nmutation = np.ceil(population_size * mutation_rate).astype(int)
    best_iter = 0
    best_route = []
    best_fit = 0
    iterator = tqdm.trange(max_iters)
    for it in iterator:
        if it == 0:
            population = np.array([np.random.permutation(n) for i in range(population_size)], dtype=int)
        else:
            idx = np.argsort(fits)
            new_population = np.zeros(population.shape, dtype=int)
            new_population[:nreserve, :] = population[idx[-nreserve:], :]
            prob = fits / np.linalg.norm(fits, ord=1)
            for i in range(ncross // 2):
                parents_idx = np.random.choice(population_size, 2, replace=False, p=prob)
                new_population[nreserve+i+i:nreserve+i+i+2, :] = cross(population[parents_idx])
            mutation_idx = np.random.choice(population_size, nmutation, replace=False)
            for i in mutation_idx:
                mutate(new_population[i])
            population = new_population
        fits = np.array([fitness(route_length(route, distance_matrix)) for route in population])
        for route, fit in zip(population, fits):
            if fit > best_fit:
                best_iter = it
                best_route = route
                best_fit = fit
        if thres > 0 and it - best_iter > thres:
            iterator.close()
            break
    return route_length(best_route, distance_matrix), best_route, it + 1


def load_tsplib(filename):
    coords = []
    with open(filename, 'r') as f:
        node_coord_section = False
        for line in f.readlines():
            if line.strip() == 'EOF':
                break
            if node_coord_section:
                x, y = line.strip().split()[1:3]
                coords.append(np.array([float(x), float(y)]))
            if line.strip() == 'NODE_COORD_SECTION':
                node_coord_section = True
    n = len(coords)
    distance_matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return np.array(coords), distance_matrix


if __name__ == '__main__':
    coords, distance_matrix = load_tsplib('./xqf131.tsp')
    length, route, it = ga(distance_matrix, population_size=64, cross_rate=0.9, mutation_rate = 0.1, max_iters=20000, thres=1000)
    print("iteration:", it)
    print("length:", length)
    print("route:", list(route))
    pts = coords[np.concatenate((route, [route[0]])), :]
    plt.plot(pts[:, 0], pts[:, 1], c='k', lw=1, marker='o', mec='r', mfc='r', ms=3)
    plt.title('iter: %d, length: %f' % (it, length))
    plt.savefig('route.pdf')
    plt.show()

