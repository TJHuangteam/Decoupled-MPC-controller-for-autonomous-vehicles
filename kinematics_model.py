import math
import numpy as np
from scipy.integrate import odeint

'''
不用修改，动力学模型
'''


class Kinematics_Model:

    def __init__(self, param):
        self.param = param
        self.L = param.L
        self.cur = 0.0

        self.v_x_0 = self.param.v_x_0
        self.v_y_0 = self.param.v_y_0  # 大地坐标系下的初始纵横向速度，用于求出加速度
        self.vX = self.v_x_0
        self.vY = self.v_y_0
        self.aX = 0.0
        self.aY = 0.0

        # 车身坐标系
        self.vx = self.v_x_0
        self.vy = 0.0

    def get_v_s(self, point, t, a):
        y = point
        return a

    def get_pos_xy_fai(self, point, t, vs, delta):
        x, y, z = point  # point为当前时刻状态点(x,y,phi)
        return np.array([vs * math.cos(z), vs * math.sin(z), vs * (math.tan(delta) / self.L)])

    def update(self, Point, sets, t):
        T = t[1] - t[0]

        delta_fc = sets[0]
        a_x = sets[1]
        v_x = odeint(self.get_v_s, Point[3], t, args=(a_x,))
        v_x = v_x[1, 0]
        Next_Point = odeint(self.get_pos_xy_fai, Point[0:3], t, args=(v_x, delta_fc))  # 解微分方程，第二个参数Point为y的初值向量

        next_point = (Next_Point[1, 0], Next_Point[1, 1], Next_Point[1, 2], v_x)
        # self.vX = v_x
        self.vX = (next_point[0] - Point[0])/T
        self.vY = (next_point[1] - Point[1]) / T
        self.aX = a_x
        self.aY = (self.vY - self.v_y_0) / T

        self.v_y_0 = self.vY
        self.cur = abs(self.aX * self.vY - self.aY * self.vX) / (self.vX ** 2 + self.vY ** 2) ** (3 / 2)

        self.vy = self.vY * math.cos(next_point[2]) - self.vX * math.sin(next_point[2])
        self.vx = next_point[3]
        return next_point
