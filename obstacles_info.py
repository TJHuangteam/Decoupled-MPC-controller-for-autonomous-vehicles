import numpy as np
from matplotlib import pyplot as plt


class ObstaclesInfo:
    def __init__(self, param):
        self.param = param
        self.t = self.param.T
        self.N = self.param.N
        self.v_x_ob = self.param.v_x_ob
        self.S_ob = np.zeros((self.param.N, 3))
        self.path = self.obstacles_path()

    def obstacles_path(self):
        self.path = np.zeros((self.param.N, 3))
        for i in range(0, self.N):
            self.path[i, 0] = i * self.t
            self.path[i, 1] = 1040 + self.v_x_ob * i * self.t
        self.path[:, 2] = self.v_x_ob

        # plt.scatter(self.path[:, 0], self.path[:, 1], marker='o', facecolor='none', edgecolors='r')
        return self.path
