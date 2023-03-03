import math
import numpy as np

'''
完成修改
'''


class Config:

    def __init__(self):
        """
        Simulation env Config
        """

        self.N = 150  # 共走多少步， 注意此时仿真步长为0.1s，假设X超过20m时开始变道，D_ref变为lane_width
        self.T = 0.05

        '''
        Vehicle Config
        '''
        self.v_x_ob = 20.0 / 3.6
        self.v_x_ref = 20.0 / 3.6
        self.v_x_0 = 35.0 / 3.6
        self.v_y_0 = 0.0 / 3.6
        self.Pos_x_0 = 1010.0

        self.a = 1.400
        self.b = 1.650
        self.L = self.a + self.b
        self.M = 1650
        self.Iz = 3234
        self.Caf = -50531 * 2
        self.Car = -43121 * 2
        self.R = 0.353  # 轮胎半径
        self.steeringratio = 16  # 方向盘转16度，前轮转1度
        self.lanewidth = 2.5  # 车道宽
        '''
        Display Config
        '''
        self.sim_result_record_enable = True


class MPC_lat_Config(Config):

    def __init__(self):
        super().__init__()
        # 参考状态量K_ref
        self.K_ref = 0
        self.Nx = 2  # State Size
        self.Nu = 1  # Input Size
        self.Ny = 2  # output size
        self.Np = 40  # 预测时域
        self.Nc = 10  # 控制时域

        self.rou_lat = 0.005  # 松弛因子


class MPC_lon_Config(Config):

    def __init__(self):
        super().__init__()

        self.dstop = 10
        self.K_ref = 0
        self.omega_ref = 0
        self.mpc_Nx = 2  # State Size
        self.mpc_Nu = 1  # Input Size
        self.mpc_Ny = 2
        self.mpc_Np = 30
        self.mpc_Nc = 10
        self.mpc_Row = 0.01
        self.mpc_Cy = [[1, 0], [0, 1]]
