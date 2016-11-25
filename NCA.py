#code by Maharsh Patel
#reference from https://github.com/RolT/NCA-python

import numpy as np
import pylab as pl
import scipy.optimize as opt


#calculates the cost of a particular point
def cost(A, X, y, threshold=None):
    (D, N) = np.shape(X)
    A = np.reshape(A, (np.size(A) / np.size(X, axis=0), np.size(X, axis=0)))
    (d, aux) = np.shape(A)
    assert D == aux

    AX = np.dot(A, X)
    normAX = np.linalg.norm(AX[:, :, None] - AX[:, None, :], axis=0)

    denomSum = np.sum(np.exp(-normAX[:, :]), axis=0)
    Pij = np.exp(- normAX) / denomSum[:, None]
    if threshold is not None:
        Pij[Pij < threshold] = 0
        Pij[Pij > 1-threshold] = 1

    mask = (y != y[:, None])
    Pijmask = np.ma.masked_array(Pij, mask)
    P = np.array(np.sum(Pijmask, axis=1))
    mask = np.negative(mask)

    f = np.sum(P)

    Xi = X[:, :, None] - X[:, None, :]
    Xi = np.swapaxes(Xi, 0, 2)

    Xi = Pij[:, :, None] * Xi

    Xij = Xi[:, :, :, None] * Xi[:, :, None, :]

    gradf = np.sum(P[:, None, None] * np.sum(Xij[:], axis=1), axis=0)

    # To optimize the distance matrix
    for i in range(N):
        aux = np.sum(Xij[i, mask[i]], axis=0)
        gradf -= aux 

    gradf = 2 * np.dot(A, gradf) #calculate the graident
    gradf = -np.reshape(gradf, np.size(gradf))
    f = np.size(X, 1) - f

    return [f, gradf]


def f(A, X, y):
    return cost(A, X, y)[0]


def grad(A, X, y):     #calculate the graident
    return cost(A, X, y)[1]



# main class which conains the functions that helps us to calculate NCA
class NCA(object):

    def __init__(self, metric=None, dim=None,threshold=None, objective='Mahalanobis', **kwargs):
        self.metric = metric
        self.dim = dim
        self.threshold = threshold
        if objective == 'Mahalanobis':
            self.objective = cost
        self.kwargs = kwargs

    def fit(self, X, y):     # function for fitting the data
        if self.metric is None:
            if self.dim is None:
                self.metric = np.eye(np.size(X, 1))
                self.dim = np.size(X, 1)
            else:
                self.metric = np.eye(self.dim, np.size(X, 1) - self.dim)

        res = opt.minimize(fun=self.objective,
                           x0=self.metric,
                           args=(X, y, self.threshold),
                           jac=True,
                           **self.kwargs
                           )

        self.metric = np.reshape(res.x,
                                 (np.size(res.x) / np.size(X, 0),
                                  np.size(X, 0)))

    def fit_transform(self, X, y):
        self.fit(self, X, y)
        return np.dot(self.metric, X)

    def score(self, X, y):
        return 1 - cost(self.metric, X, y)[0]/np.size(X, 1)

    def getParameters(self):
        return dict(metric=self.metric, dim=self.dim, objective=self.objective,
                     threshold=self.threshold, **self.kwargs)


# Initialisation with no of samples equal to 700
N = 700
aux = (np.concatenate([0.5*np.ones((N/2, 1)),
                       np.zeros((N/2, 1)), 1.1*np.ones((N/2, 1))], axis=1))
X = np.concatenate([np.random.rand(N/2, 3),
                    np.random.rand(N/2, 3) + aux])

y = np.concatenate([np.concatenate([np.ones((N/2, 1)), np.zeros((N/2, 1))]),
                    np.concatenate([np.zeros((N/2, 1)), np.ones((N/2, 1))])],
                   axis=1)
X = X.T
y = y[:, 0]
A = np.array([[1, 0, 0], [0, 1, 0]])

# Training the data-set
nca = NCA(metric=A, method='BFGS', objective='Mahalanobis', options={'maxiter': 10, 'disp': True})
print nca.score(X, y)
nca.fit(X, y)
print nca.score(X, y)

# Plot the graph
pl.subplot(2, 1, 1)
AX = np.dot(A, X)
pl.scatter(AX[0, :], AX[1, :], 30, c=y,cmap=pl.cm.Paired)
pl.subplot(2, 1, 2)
BX = np.dot(np.reshape(nca.metric, np.shape(A)), X)
pl.scatter(BX[0, :], BX[1, :], 30, c=y,cmap=pl.cm.Paired)
pl.show()
