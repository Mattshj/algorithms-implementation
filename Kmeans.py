from copy import deepcopy

import numpy as np
import random as rd
from numpy import linalg as LA, math
# from random import random

class Kmeans:
    def __init__(self, D, K):
        self.D = D
        self.K = K
        centers1=[]
        for i in range(K):

            centers1.append(D[rd.randint(0, D.shape[0])])

        centers2=np.array(centers1)
        self.centers = deepcopy(centers2)
        self.dist = np.zeros((D.shape[0], K))
        self.clusters = np.zeros(D.shape[0])


    def fit(self, iter):

        for i in range(iter):
            for i in range(self.K):
                self.dist[:, i] = LA.norm(self.D - self.centers[i], axis=1)
            self.clusters = np.argmin(self.dist, axis=1)

            # self.centers_new = np.array([np.mean(self.D[self.clusters == k], axis=0) for k in range(self.K) if len(np.where(self.clusters == i)[0]) != 0])
            for i in range(self.K):
                if len(np.where(self.clusters == i)[0]) != 0:
                    self.centers[i] = np.mean(self.D[self.clusters == i], axis=0)
        return self.centers,self.clusters
