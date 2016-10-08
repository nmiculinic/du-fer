import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class Random2DGaussian:

    def __init__(self, minx=0, maxx=10, miny=0, maxy=10):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

        self.mu = np.array([
            np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
        ])

        eigvalx = (np.random.random_sample() * (maxx - minx) / 5)**2
        eigvaly = (np.random.random_sample() * (maxy - miny) / 5)**2
        D = np.diag([eigvalx, eigvaly])

        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

        self.sigma = np.dot(np.dot(R.T, D), R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)


def sample_gauss_2d(C, N):
    X, Y = [], []
    for i in range(C):
        G = Random2DGaussian()
        X.append(G.get_sample(N))
        Y.append(np.full((N, 1), i, dtype=np.int32))
    X = np.vstack(X)
    Y = np.vstack(Y)
    return X, Y


def graph_data(X, Y_, Y):
    '''
    X  ... podatci (np.array dimenzija Nx2)
    Y_ ... točni indeksi razreda podataka (Nx1)
    Y  ... predviđeni indeksi razreda podataka (Nx1)
    '''

    fig = plt.figure()
    T = (Y == Y_).reshape(-1)
    F = ~T
    Xt = X[T]
    Xf = X[F]

    norm = mpl.colors.Normalize()
    norm.autoscale(Y_)
    plt.scatter(Xt[:, 0], Xt[:, 1], c=Y_[T], marker='o', norm=norm, cmap='Paired')
    plt.scatter(Xf[:, 0], Xf[:, 1], c=Y_[F], marker='s', norm=norm, cmap='Paired')
    return fig
