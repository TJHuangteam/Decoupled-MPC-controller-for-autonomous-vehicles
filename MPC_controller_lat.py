import numpy as np
import math
from scipy import sparse
from cvxopt import matrix, solvers


class MPC_controller_lat:

    def __init__(self, path, param, model):

        self.param = param

        self.D_ref = path
        self.Nx = self.param.Nx
        self.Nu = self.param.Nu
        self.Ny = self.param.Ny
        self.lanewidth = self.param.lanewidth
        self.Np = self.param.Np
        self.Nc = self.param.Nc
        self.T = self.param.T

        self.a = self.param.a
        self.b = self.param.b
        self.M = self.param.M
        self.Iz = self.param.Iz
        self.Caf = self.param.Caf
        self.Car = self.param.Car
        self.R = self.param.R
        self.steeringratio = self.param.steeringratio
        self.K_ref = self.param.K_ref
        '''
        把车辆当前参数引进来
        '''

        self.rou_lat = self.param.rou_lat  # rho的值

        eps = 1e-10

        # 硬约束边界
        self.K_min = -30 / 180 * np.pi  # 航向角差的约束
        self.K_max = -self.K_min

        self.D_min = -self.lanewidth / 2  # Frenet坐标系下相对参考线横向偏差的约束
        self.D_max = self.lanewidth + 1

        self.vD_min = -5 / 3.6  # 横向速度约束
        self.vD_max = -self.vD_min

        '''
        横摆角约束要与当前的后轴速度有关，因此放在动态计算中
        下面的代码中先给出一个与速度无关的约束，即只要求转角在±90度内
        omega_min = -5 / 180 * np.pi    #横摆角约束
        omega_max = -omega_min

        self.deg1 = np.arctan(omega_min / (vx + eps) * (self.a + self.b))    #v应该是后轴速度，运动学模型情况下近似为vx。
        self.deg2 = np.arctan(omega_max / (vx + eps) * (self.a + self.b))
        self.delta_f_min = max(-90 / (self.steeringratio + eps) / 180 * np.pi, self.deg1)   #前轮转角的约束，令方向盘正反打90度为界，与用w求出的界进行比较，选出更合适的
        self.delta_f_max = min(90 / (self.steeringratio + eps) / 180 * np.pi, self.deg2)
        '''
        self.delta_f_min = -90 / (self.steeringratio + eps) / 180 * np.pi  # 前轮转角的约束，令方向盘正反打90度为界，与用w求出的界进行比较，选出更合适的
        self.delta_f_max = 90 / (self.steeringratio + eps) / 180 * np.pi

        self.d_delta_f_min = (-400 / (self.steeringratio) / 180 * np.pi) * 0.05  # delta_U的约束
        self.d_delta_f_max = -self.d_delta_f_min
        self.e_min_lat = 0  # 松弛因子的约束
        self.e_max_lat = 0.05

        # 约束矩阵
        self.x_max_ext_lat = np.zeros([self.Np * self.Nx, 1])
        self.x_min_ext_lat = np.zeros([self.Np * self.Nx, 1])
        self.y_max_ext_lat = np.zeros([self.Np * self.Ny, 1])
        self.y_min_ext_lat = np.zeros([self.Np * self.Ny, 1])
        self.u_max_ext_lat = np.zeros([self.Nc * self.Nu, 1])
        self.u_min_ext_lat = np.zeros([self.Nc * self.Nu, 1])
        self.du_max_ext_lat = np.zeros([self.Nc * self.Nu, 1])
        self.du_min_ext_lat = np.zeros([self.Nc * self.Nu, 1])
        for i in range(self.Np):
            self.x_max_ext_lat[i * self.Nx: (i + 1) * self.Nx] = np.array([[self.K_max], [self.D_max]])
            self.x_min_ext_lat[i * self.Nx: (i + 1) * self.Nx] = np.array([[self.K_min], [self.D_min]])
            self.y_max_ext_lat[i * self.Nx: (i + 1) * self.Nx] = np.array([[self.K_max], [self.D_max]])
            self.y_min_ext_lat[i * self.Nx: (i + 1) * self.Nx] = np.array([[self.K_min], [self.D_min]])
            if i < self.Nc:
                self.u_max_ext_lat[i * self.Nu: (i + 1) * self.Nu] = np.array([[self.delta_f_max]])
                self.u_min_ext_lat[i * self.Nu: (i + 1) * self.Nu] = np.array([[self.delta_f_min]])
                self.du_max_ext_lat[i * self.Nu: (i + 1) * self.Nu] = np.array([[self.d_delta_f_max]])
                self.du_min_ext_lat[i * self.Nu: (i + 1) * self.Nu] = np.array([[self.d_delta_f_min]])

        # 权重矩阵
        self.q_lat = 1 * 10 * np.diag([1 / self.K_max ** 2, 1 / (self.lanewidth + eps) ** 2])
        # q_lat_last = 1 * q_lat
        self.ru_lat = 1 / 100 * np.diag([1 / (self.delta_f_max + eps) ** 2])  # 控制量的权重矩阵
        '''
        用到动态量D，因此要放在动态计算中，下方先给出一个静态约束
        # if D >= 1.75:
        #     self.rdu_lat = 0 / 100000 * np.diag([1 / (self.d_delta_f_max + eps) ** 2])     #控制量增量的权重矩阵，当偏离很大时，设为0，让他可以快速调节，保证安全性
        # else:
        #     self.rdu_lat = 1 / 100000 * np.diag([1 / (self.d_delta_f_max + eps) ** 2])   #偏离不大时，让他微调，保证舒适性
        '''

        self.rdu_lat = 1 / 100000 * np.diag([1 / (self.d_delta_f_max + eps) ** 2])

        # 预测时域和控制时域内的分块权重矩阵
        Q_cell = np.empty((self.Np, self.Np), dtype=object)
        for i in range(self.Np):
            for j in range(self.Np):
                Q_cell[i, j] = self.q_lat if i == j else np.zeros_like(self.q_lat)
        self.Q_lat = np.vstack([np.hstack(row) for row in Q_cell])

        Ru_cell = np.empty((self.Nc, self.Nc), dtype=object)
        for i in range(self.Nc):
            for j in range(self.Nc):
                Ru_cell[i, j] = self.ru_lat if i == j else np.zeros_like(self.ru_lat)
        self.Ru_lat = np.vstack([np.hstack(row) for row in Ru_cell])

        Rdu_cell = np.empty((self.Nc, self.Nc), dtype=object)
        for i in range(self.Nc):
            for j in range(self.Nc):
                Rdu_cell[i, j] = self.rdu_lat if i == j else np.zeros_like(self.rdu_lat)
        self.Rdu_lat = np.vstack([np.hstack(row) for row in Rdu_cell])

        # 存放输入量
        # self.u = np.zeros((self.Nu, 1))         #真实控制量与参考控制量的差
        self.u_real = np.zeros((self.Nu, 1))  # 真实控制量

    def f(self, x_current, u_last, a, b, vx, cur, vy, vS):
        K = x_current[0, 0]
        D = x_current[1, 0]
        delta_f = u_last

        Matrix = np.array([[(vx * math.tan(delta_f)) / (a + b) - cur * vS],
                           [vy * math.cos(K) + vx * math.sin(K)]])

        return Matrix

    def Jf_x(self, x_current, u_last, a, b, vx, cur, vy, vS):
        K = x_current[0, 0]
        D = x_current[1, 0]
        delta_f = u_last

        '''
        #vS也是vx和vy和K的表达的，此处偏导是否存在问题？  代入换一下
        Matrix = np.array([[cur*(vx * math.sin(K)+vy*math.cos(K)/(1-D*cur)), cur*cur*(vx * math.cos(K) - vy * math.sin(K))/(1-D*cur)**2],
                           [vx * math.cos(K) - vy * math.sin(K), 0]])
        '''
        Matrix = np.array([[0, 0],
                           [vx * math.cos(K) - vy * math.sin(K), 0]])

        return Matrix

    def Jf_u(self, x_current, u_last, a, b, vx):
        K = x_current[0, 0]
        D = x_current[1, 0]
        delta_f = u_last

        Matrix = np.array([[(vx * (math.tan(delta_f) ** 2 + 1)) / (a + b)],
                           [0.]])
        return Matrix

    def calc_input(self, D_ref, vx, vy, x_current_lat, u_lat_last, cur):

        D = x_current_lat[1, 0]
        K = x_current_lat[0, 0]
        vS = (vx * math.cos(K) - vy * math.sin(K)) / (1 - D * cur)
        # X_current_lat是列向量，其他的是标量，vx为车身坐标系下纵向车身速度(后轴速度)，vy为车身坐标系下横向速度，vS为frenet下纵向速度

        # 计算参考轨迹
        y_ref_ext_lat = np.zeros([self.Ny * self.Np, 1])  # 要跟随的参考观测量
        for i in range(self.Np):
            if D_ref >= D:  # 如果要跟随的目标在当前位置的外侧
                D_ref0 = D + (i + 1) * ((D_ref - D) / (self.Np))  # 类似于按照等间隔从D增加到D_ref,按照10m/s的速度正好加到D_ref
            else:
                D_ref0 = D_ref + (self.Np - i) * ((D - D_ref) / (self.Np))

            y_ref_ext_lat[i * self.Ny: (i + 1) * self.Ny, 0] = [self.K_ref, D_ref0]  # K_ref一直为0，理想状况

        #####################################模型线性化#######################################################
        '''
        #当纵向速度会变时用这个
        vx_cell = np.zeros([self.Np, 1])
        for i in range(self.Np):
            if i == 1:
                vx_cell[i] = vx
            else:
                vx_cell[i] = vx_cell[i - 1] + 0 * self.T      #意义何在？  预测的Np步的每步vx
        '''
        vx_cell = vx * np.ones([self.Np, 1])

        # 假设控制量不变，来预测车辆的未来轨迹作为参考轨迹
        SV_r_cell = np.zeros([self.Nx * self.Np, 1])  # 保存施加不变的控制量下预测的每个时刻的状态量，作为参考轨迹X_r，用来进行线性化;u_r为上一时刻输入不变
        for i in range(self.Np):  # 使用微分方程格式计算未来时刻的状态量
            if i == 0:
                SV_r_cell[i * self.Nx: (i + 1) * self.Nx] = x_current_lat + self.f(x_current_lat, u_lat_last, self.a,
                                                                                   self.b, vx_cell[i, 0], cur, vy,
                                                                                   vS) * self.T
                #   为何vS处也要代入vx？？   大多数情况下vs和vx可以互换，因为摆动不剧烈，为了运算方便；也可以改成vs。
            else:
                SV_r_cell[i * self.Nx: (i + 1) * self.Nx] = SV_r_cell[(i - 1) * self.Nx: i * self.Nx] + self.f(
                    SV_r_cell[(i - 1) * self.Nx: i * self.Nx], u_lat_last, self.a, self.b, vx_cell[i, 0], cur, vy,
                    vx_cell[i, 0]) * self.T

        # 计算预测时域内k时刻的线性化矩阵，x(k+1)=A(k)x(k)+B(k)u(k)+Ck
        A_cell_lat = np.zeros([self.Nx, self.Nx, self.Np])  # 2 * 2 * Np的矩阵，第三个维度为每个时刻的对应的A矩阵
        B_cell_lat = np.zeros([self.Nx, self.Nu, self.Np])  # 2 * 1 * Np的矩阵
        C_cell_lat = np.zeros([self.Nx, self.Np])  # 2 * Np的矩阵

        for i in range(self.Np):  # 保存每个预测时间步的Ak，Bk，Ck矩阵
            # Ak = I + T * J_x
            A_cell_lat[:, :, i] = np.eye(self.Nx) + self.T * self.Jf_x(SV_r_cell[i * self.Nx: (i + 1) * self.Nx],
                                                                       u_lat_last, self.a, self.b, vx_cell[i, 0], cur,
                                                                       vy, vx_cell[i, 0])
            # Bk = T * J_u
            B_cell_lat[:, :, i] = self.T * self.Jf_u(SV_r_cell[i * self.Nx: (i + 1) * self.Nx], u_lat_last, self.a,
                                                     self.b, vx_cell[i, 0])
            # 应该乘i时刻的x_r而不是起始位置x_current??   改一下
            C_cell_lat[:, i:i + 1] = self.T * self.f(SV_r_cell[i * self.Nx: (i + 1) * self.Nx], u_lat_last, self.a,
                                                     self.b, vx_cell[i, 0], cur, vy, vx_cell[i, 0]) - (
                                             A_cell_lat[:, :, i] - np.eye(self.Nx)) @ x_current_lat - B_cell_lat[:,
                                                                                                      :,
                                                                                                      i] * u_lat_last

        #################################################动态矩阵计算########################################################
        A_ext_lat = np.zeros([self.Nx * self.Np, self.Nx])  # 2Np * 2的分块列矩阵
        B_ext_lat = np.zeros([self.Nx * self.Np, self.Nu * self.Nc])  # 2Np * Nc的分块矩阵
        for i in range(self.Np):  # 递推形式下的A_bar矩阵
            if i == 0:
                A_ext_lat[i * self.Nx: (i + 1) * self.Nx, :] = A_cell_lat[:, :, i]
            else:
                A_ext_lat[i * self.Nx: (i + 1) * self.Nx, :] = A_cell_lat[:, :, i] @ A_ext_lat[
                                                                                     (i - 1) * self.Nx: i * self.Nx, :]

        for i in range(self.Np):
            if i < self.Nc:  # 对角元
                B_ext_lat[i * self.Nx: (i + 1) * self.Nx, i * self.Nu: (i + 1) * self.Nu] = B_cell_lat[:, :,
                                                                                            i]  # 维数相同，每个小块是Nx*Nu维度
            else:  # Nc行之后的最后一列元素，形式改变了，先给了一个Bk(k > Nc)
                B_ext_lat[i * self.Nx: (i + 1) * self.Nx, (self.Nc - 1) * self.Nu: self.Nc * self.Nu] = B_cell_lat[:, :,
                                                                                                        i]

        for i in range(1, self.Np):  # 次对角元素，每一行都是上一行乘新的Ai，再加上上一步已经加好的每行最后一个元素，就得到B_bar
            B_ext_lat[i * self.Nx: (i + 1) * self.Nx, :] = A_cell_lat[:, :, i] @ B_ext_lat[
                                                                                 (i - 1) * self.Nx: i * self.Nx,
                                                                                 :] + B_ext_lat[
                                                                                      i * self.Nx: (i + 1) * self.Nx, :]

        C_ext_lat = np.zeros([self.Nx * self.Np, 1])  # 2Np * 1的矩阵
        for i in range(self.Np):
            if i == 0:
                C_ext_lat[i * self.Nx: (i + 1) * self.Nx] = C_cell_lat[:, i:i + 1]
            else:  # C的上一行乘以新的Ak再加上新的Ck得到C_bar
                C_ext_lat[i * self.Nx: (i + 1) * self.Nx] = A_cell_lat[:, :, i] @ C_ext_lat[(
                                                                                                    i - 1) * self.Nx: i * self.Nx] + C_cell_lat[
                                                                                                                                     :,
                                                                                                                                     i:i + 1]  # 写成[:,i:i+1]能够保证都为二维数组

        Cy_ext = np.eye(self.Np * self.Ny)  # 观测矩阵，2Np * 2Np，是一个单位阵
        Q = self.Q_lat  # 误差的大权重矩阵，2Np * 2Np，对角阵
        Ru = self.Ru_lat  # 状态量的大权重矩阵，2Np * 2Np，对角阵
        Rdu = self.Rdu_lat  # 状态量增量的大权重矩阵，2Np * 2Np，对角阵

        # 根据U矩阵得到deltaU矩阵，deltaU = Cdu * U + Du
        # Cdu矩阵
        Cdu1_lat = np.empty((2, 2), dtype=object)
        Cdu1_lat[0, 0] = np.zeros([self.Nu, self.Nu * (self.Nc - 1)])
        Cdu1_lat[0, 1] = np.zeros([self.Nu, self.Nu])
        Cdu1_lat[1, 0] = -1 * np.eye(self.Nu * (self.Nc - 1))
        Cdu1_lat[1, 1] = np.zeros([self.Nu * (self.Nc - 1), self.Nu])
        Cdu1_lat = np.vstack([np.hstack(Mat_size) for Mat_size in Cdu1_lat])
        Cdu1_lat = Cdu1_lat + np.eye(self.Nu * self.Nc)

        # Cdu1_lat = [np.zeros(n_lat, Nc_lat * n_lat); -eye((Nc_lat - 1) * n_lat), zeros((Nc_lat - 1) * n_lat,
        # n_lat)] + eye(Nc_lat * n_lat) Du矩阵（列向量）
        Cdu2_lat = np.zeros([self.Nu * self.Nc, 1])
        # 为了让多维输入也可以通用
        Cdu2_lat[0 * self.Nu:1 * self.Nu, 0] = -1 * u_lat_last
        # Cdu2_lat[0,0] = -1* u_lat_last
        # Cdu2_lat = [-u_lat_last; zeros((Nc_lat - 1) * n_lat, 1)]

        ########################QP矩阵计算，控制量为[du',e']'##############################################
        # 标准形式min (1/2x'Px+q'x)   s.t. Gx<=h
        H_QP_du_e_lat = np.empty((2, 2), dtype=object)
        H_QP_du_e_lat[0, 0] = np.transpose(Cy_ext @ B_ext_lat @ np.linalg.inv(Cdu1_lat)) @ Q @ (
                Cy_ext @ B_ext_lat @ np.linalg.inv(Cdu1_lat)) + np.transpose(
            np.linalg.inv(Cdu1_lat)) @ Ru @ np.linalg.inv(Cdu1_lat) + Rdu
        H_QP_du_e_lat[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        H_QP_du_e_lat[1, 0] = np.zeros([1, self.Nc * self.Nu])
        H_QP_du_e_lat[1, 1] = self.rou_lat * np.eye(1)
        H_QP_du_e_lat = np.vstack([np.hstack(Mat_size) for Mat_size in H_QP_du_e_lat])
        H_QP_du_e_lat = 2 * H_QP_du_e_lat
        # H_QP_du_e_lat = 2 * [(Cy_ext * B_ext_lat * inv(Cdu1_lat))'*Q*(Cy_ext*B_ext_lat*inv(Cdu1_lat))+inv(Cdu1_lat)' * Ru * inv(Cdu1_lat) + Rdu,zeros(Nc_lat * n_lat, 1);zeros(1, Nc_lat * n_lat), rou_lat];

        # q为列向量
        f_QP_du_e_lat = np.empty((1, 2), dtype=object)
        f_QP_du_e_lat[0, 0] = 2 * np.transpose(Cy_ext @ (A_ext_lat @ x_current_lat - B_ext_lat @ np.linalg.inv(
            Cdu1_lat) @ Cdu2_lat + C_ext_lat) - y_ref_ext_lat) @ Q @ Cy_ext @ B_ext_lat @ np.linalg.inv(
            Cdu1_lat) + 2 * np.transpose(np.linalg.inv(Cdu1_lat) @ (-Cdu2_lat)) @ Ru @ np.linalg.inv(Cdu1_lat)
        f_QP_du_e_lat[0, 1] = np.zeros([1, 1])
        f_QP_du_e_lat = np.vstack([np.hstack(Mat_size) for Mat_size in f_QP_du_e_lat])
        f_QP_du_e_lat = f_QP_du_e_lat.T  # 转置
        # f_QP_du_e_lat = np.transpose(f_QP_du_e_lat) f_QP_du_e_lat = [2 * (Cy_ext * (A_ext_lat * x_current_lat +
        # B_ext_lat * inv(Cdu1_lat) * (-Cdu2_lat) + C_ext_lat) - y_ref_ext_lat)'*Q*Cy_ext*B_ext_lat*inv(Cdu1_lat)+2*(
        # inv(Cdu1_lat)*(-Cdu2_lat))' * Ru * inv(Cdu1_lat), 0]'

        A_du_eCons_lat = np.empty([6, 2], dtype=object)
        ubA_du_eCons_lat = np.empty([6, 1], dtype=object)
        A_du_eCons_lat[0, 0] = np.linalg.inv(Cdu1_lat)
        A_du_eCons_lat[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons_lat[1, 0] = B_ext_lat @ np.linalg.inv(Cdu1_lat)
        A_du_eCons_lat[1, 1] = np.zeros([self.Nx * self.Np, 1])
        A_du_eCons_lat[2, 0] = Cy_ext @ B_ext_lat @ np.linalg.inv(Cdu1_lat)
        A_du_eCons_lat[2, 1] = np.zeros([self.Np * self.Ny, 1])
        A_du_eCons_lat[3, 0] = -np.linalg.inv(Cdu1_lat)
        A_du_eCons_lat[3, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons_lat[4, 0] = -B_ext_lat @ np.linalg.inv(Cdu1_lat)
        A_du_eCons_lat[4, 1] = np.zeros([self.Nx * self.Np, 1])
        A_du_eCons_lat[5, 0] = -Cy_ext @ B_ext_lat @ np.linalg.inv(Cdu1_lat)
        A_du_eCons_lat[5, 1] = np.zeros([self.Np * self.Ny, 1])
        # A_du_eCons_lat[6,0] = np.eye(self.Nc*self.Nu)
        # A_du_eCons_lat[6,1] = np.zeros([self.Nc*self.Nu,1])
        # A_du_eCons_lat[7,0] = np.zeros([1,self.Nc*self.Nu])
        # A_du_eCons_lat[7,1] = np.array([[1.]])
        # A_du_eCons_lat[8,0] = -np.eye(self.Nc,self.Nu)
        # A_du_eCons_lat[8,1] = np.zeros([self.Nc*self.Nu,1])
        # A_du_eCons_lat[9,0] = np.zeros([1,self.Nc*self.Nu])
        # A_du_eCons_lat[9,1] = np.array([[-1.]])
        A_du_eCons_lat = np.vstack([np.hstack(Mat_size) for Mat_size in A_du_eCons_lat])

        ubA_du_eCons_lat[0, 0] = self.u_max_ext_lat + Cdu1_lat @ Cdu2_lat
        ubA_du_eCons_lat[1, 0] = self.x_max_ext_lat - A_ext_lat @ x_current_lat - C_ext_lat + B_ext_lat @ np.linalg.inv(
            Cdu1_lat) @ Cdu2_lat
        ubA_du_eCons_lat[2, 0] = self.y_max_ext_lat - Cy_ext @ (
                A_ext_lat @ x_current_lat + C_ext_lat - B_ext_lat @ np.linalg.inv(Cdu1_lat) @ Cdu2_lat)
        ubA_du_eCons_lat[3, 0] = -self.u_min_ext_lat - Cdu1_lat @ Cdu2_lat
        ubA_du_eCons_lat[
            4, 0] = -self.x_min_ext_lat + A_ext_lat @ x_current_lat + C_ext_lat - B_ext_lat @ np.linalg.inv(
            Cdu1_lat) @ Cdu2_lat
        ubA_du_eCons_lat[5, 0] = -self.y_min_ext_lat + Cy_ext @ (
                A_ext_lat @ x_current_lat + C_ext_lat - B_ext_lat @ np.linalg.inv(Cdu1_lat) @ Cdu2_lat)
        # ubA_du_eCons_lat[6,0] = self.du_max_ext_lat
        # ubA_du_eCons_lat[7,0] = self.e_max_lat * np.eye(1)
        # ubA_du_eCons_lat[8,0] = -self.du_min_ext_lat
        # ubA_du_eCons_lat[9,0] = -self.e_min_lat * np.eye(1)
        ubA_du_eCons_lat = np.vstack([np.hstack(Mat_size) for Mat_size in ubA_du_eCons_lat])
        lb = np.vstack((self.du_min_ext_lat, self.e_min_lat))  # 使用vstack堆叠数组
        ub = np.vstack((self.du_max_ext_lat, self.e_max_lat))

        P = matrix(H_QP_du_e_lat)
        q = matrix(f_QP_du_e_lat)
        G = matrix(np.vstack((A_du_eCons_lat, np.eye(self.Nc * self.Nu + 1), -np.eye(self.Nc * self.Nu + 1))))
        h = matrix(np.vstack((ubA_du_eCons_lat, ub, -lb)))

        result = solvers.qp(P, q, G, h)  # 1/2x'Px+q'x   Gx<=h  Ax=b 注意使用qp时，每个参数要换成matrix
        # 重要：print可被关闭
        X = result['x']  # 'x'为result中的解，'status'表示是否找到最优解。

        u_incr = X[0]  # 控制量增量
        delta_fc = u_lat_last + u_incr

        Input = delta_fc

        return Input
