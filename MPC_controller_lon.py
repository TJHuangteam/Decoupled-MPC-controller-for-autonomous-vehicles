import numpy as np
from cvxopt import matrix, solvers


class MPC_controller_lon:

    def __init__(self, path, param, model):

        self.path = path
        self.model = model
        self.param = param
        self.T = self.param.T
        self.N = self.param.N
        self.Nx = self.param.mpc_Nx
        self.Nu = self.param.mpc_Nu
        self.Ny = self.param.mpc_Ny
        self.Np = self.param.mpc_Np
        self.Nc = self.param.mpc_Nc
        self.Cy_lon = self.param.mpc_Cy

        self.dstop = self.param.dstop
        self.v_x_ref = self.param.v_x_ref
        self.v_x_0 = self.param.v_x_0
        self.Pos_x_0 = self.param.Pos_x_0
        self.u_lon_last = 0

        # 纵向约束
        self.vS_min = 0.0
        self.vS_max = 50 / 3.6
        self.aS_min = -2 / 1
        self.aS_max = 1 / 1
        self.delta_aS_min = -20 / 1 * self.T
        # self.delta_aS_max = 2 / 1 * self.T / 10
        self.delta_aS_max = 20 / 1 * self.T
        self.e_min_lon = 0  # 松弛因子的约束
        self.e_max_lon = 0.05

        # 权重矩阵
        self.q_lon = 1 * np.diag([1 / 1 ** 2, 0 / 1 ** 2])
        self.ru_lon = 0 * np.diag([1 / (1 ** 2)])  # 控制量的权重矩阵
        self.rdu_lon = 10 * np.diag([1 / self.delta_aS_max ** 2])
        self.rou_lon = 0 * 0.005 * 1  # rho的值

        # 约束矩阵
        self.u = np.zeros((self.Nu, 1))
        self.x_max_ext_lon = np.zeros([self.Np * self.Nx, 1])
        self.x_min_ext_lon = np.zeros([self.Np * self.Nx, 1])
        self.y_max_ext_lon = np.zeros([self.Np * self.Ny, 1])
        self.y_min_ext_lon = np.zeros([self.Np * self.Ny, 1])
        self.u_max_ext_lon = np.zeros([self.Nc * self.Nu, 1])
        self.u_min_ext_lon = np.zeros([self.Nc * self.Nu, 1])
        self.du_max_ext_lon = np.zeros([self.Nc * self.Nu, 1])
        self.du_min_ext_lon = np.zeros([self.Nc * self.Nu, 1])
        self.S_max = np.zeros((self.Np, 1))
        self.S_min = np.zeros((self.Np, 1))
        self.S_ref = np.zeros((self.Np, 1))
        self.y_ref_ext_lon = np.zeros([self.Np * self.Ny, 1])

    def calc_input(self, S_obj, v_obj, x_current_lon, u_lon_last, time_counts):

        self.u_lon_last = u_lon_last

        for i in range(self.Np):
            self.S_max[i] = S_obj + v_obj * (i + 1) * self.T
            self.S_min[i] = -980
            self.S_ref[i] = self.S_max[i] - self.dstop

        for i in range(self.Np):
            self.y_ref_ext_lon[i * self.Ny: (i + 1) * self.Ny, :] = np.array([[self.S_ref[i, 0]], [self.v_x_ref]])

        for i in range(self.Np):
            self.x_max_ext_lon[i * self.Nx: (i + 1) * self.Nx, :] = np.array([[self.S_max[i, 0]], [self.vS_max]])
            self.x_min_ext_lon[i * self.Nx: (i + 1) * self.Nx, :] = np.array([[self.S_min[i, 0]], [self.vS_min]])
            self.y_max_ext_lon[i * self.Ny: (i + 1) * self.Ny, :] = np.array([[self.S_max[i, 0]], [self.vS_max]])
            self.y_min_ext_lon[i * self.Ny: (i + 1) * self.Ny, :] = np.array([[self.S_min[i, 0]], [self.vS_min]])

            if i < self.Nc:
                self.u_max_ext_lon[i] = self.aS_max
                self.u_min_ext_lon[i] = self.aS_min
                self.du_min_ext_lon[i] = self.delta_aS_min
                self.du_max_ext_lon[i] = self.delta_aS_max

        # model linearization
        Ap = np.array([[1, self.T], [0, 1]])
        Bp = np.array([[0], [self.T]])
        Cp = np.array([[0], [0]])

        A_cell_lon = np.zeros([self.Nx, self.Nx, self.Np])  # 2 * 2 * Np的矩阵，第三个维度为每个时刻的对应的A矩阵
        B_cell_lon = np.zeros([self.Nx, self.Nu, self.Np])  # 2 * 1 * Np的矩阵
        C_cell_lon = np.zeros([self.Nx, self.Np])  # 2 * Np的矩阵

        for i in range(self.Np):  # 保存每个预测时间步的Ak，Bk，Ck矩阵
            A_cell_lon[:, :, i] = Ap
            B_cell_lon[:, :, i] = Bp
            C_cell_lon[:, i:i+1] = Cp[:, 0:1]

        # dynamicmatrix:
        A_ext_lon = np.zeros([self.Nx * self.Np, self.Nx])  # 2Np * 2的分块列矩阵
        B_ext_lon = np.zeros([self.Nx * self.Np, self.Nu * self.Nc])  # 2Np * Nc的分块矩阵
        for i in range(self.Np):  # 递推形式下的A_bar矩阵
            if i == 0:
                A_ext_lon[i * self.Nx: (i + 1) * self.Nx, :] = A_cell_lon[:, :, i]
            else:
                A_ext_lon[i * self.Nx: (i + 1) * self.Nx, :] = A_cell_lon[:, :, i] @ A_ext_lon[(i - 1) * self.Nx:
                                                                                               i * self.Nx,
                                                                                     :]

        for i in range(self.Np):
            if i < self.Nc:  # 对角元
                B_ext_lon[i * self.Nx: (i + 1) * self.Nx, i * self.Nu: (i + 1) * self.Nu] = B_cell_lon[:, :,
                                                                                            i]  # 维数相同，每个小块是Nx*Nu维度
            else:  # Nc行之后的最后一列元素，形式改变了，先给了一个Bk(k > Nc)
                B_ext_lon[i * self.Nx: (i + 1) * self.Nx, (self.Nc - 1) * self.Nu: self.Nc * self.Nu] = B_cell_lon[:, :,
                                                                                                        i]

        for i in range(1, self.Np):  # 次对角元素，每一行都是上一行乘新的Ai，再加上上一步已经加好的每行最后一个元素，就得到B_bar
            B_ext_lon[i * self.Nx: (i + 1) * self.Nx, :] = A_cell_lon[:, :, i] @ B_ext_lon[
                                                                                 (i - 1) * self.Nx: i * self.Nx,
                                                                                 :] + B_ext_lon[
                                                                                      i * self.Nx: (i + 1) * self.Nx, :]

        C_ext_lon = np.zeros([self.Nx * self.Np, 1])  # 2Np * 1的矩阵
        for i in range(self.Np):
            if i == 0:
                C_ext_lon[i * self.Nx: (i + 1) * self.Nx] = C_cell_lon[:,i:i+1]
            else:  # C的上一行乘以新的Ak再加上新的Ck得到C_bar
                C_ext_lon[i * self.Nx: (i + 1) * self.Nx] = A_cell_lon[:, :, i] @ C_ext_lon[
                                                                                     (i - 1) * self.Nx: i * self.Nx] + C_cell_lon[:, i:i+1]

        # 预测时域和控制时域内的分块权重矩阵
        Cy_ext = np.zeros([self.Np * self.Ny, self.Np * self.Nx])
        Q_cell = np.zeros([self.Np * self.Ny, self.Np * self.Ny])
        Ru_cell = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])
        Rdu_cell = np.zeros([self.Nc * self.Nu, self.Nc * self.Nu])
        for i in range(self.Np):
            Cy_ext[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.Cy_lon
        for i in range(self.Np - 2):
            Q_cell[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.q_lon
        for i in range(self.Np - 2, self.Np):
            Q_cell[i * self.Ny:(i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.q_lon
        for i in range(self.Nc - 1):
            Ru_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.ru_lon
        for i in range(self.Nc - 1 + 1, self.Nc):
            Ru_cell[i * self.Nu: (i + 1) * self.Nu, i * self.Nu: (i + 1) * self.Nu] = self.ru_lon
        for i in range(self.Nc):
            Rdu_cell[i * self.Ny: (i + 1) * self.Ny, i * self.Ny: (i + 1) * self.Ny] = self.rdu_lon

        # 根据U矩阵得到deltaU矩阵，deltaU = Cdu * U + Du
        # Cdu矩阵
        Cdu1_lon = np.empty((2, 2), dtype=object)
        Cdu1_lon[0, 0] = np.zeros([self.Nu, self.Nu * (self.Nc - 1)])
        Cdu1_lon[0, 1] = np.zeros([self.Nu, self.Nu])
        Cdu1_lon[1, 0] = -1 * np.eye(self.Nu * (self.Nc - 1))
        Cdu1_lon[1, 1] = np.zeros([self.Nu * (self.Nc - 1), self.Nu])
        Cdu1_lon = np.vstack([np.hstack(Mat_size) for Mat_size in Cdu1_lon])
        Cdu1_lon = Cdu1_lon + np.eye(self.Nu * self.Nc)

        # Du矩阵（列向量）
        Cdu2_lon = np.zeros([self.Nu * self.Nc, 1])
        Cdu2_lon[0 * self.Nu:1 * self.Nu, 0] = -1 * self.u_lon_last

        # 标准形式min (1/2x'Px+q'x)   s.t. Gx<=h
        H_QP_du_e_lon = np.empty((2, 2), dtype=object)
        H_QP_du_e_lon[0, 0] = np.transpose(Cy_ext @ B_ext_lon @ np.linalg.inv(Cdu1_lon)) @ Q_cell @ (
                Cy_ext @ B_ext_lon @ np.linalg.inv(Cdu1_lon)) + np.transpose(
            np.linalg.inv(Cdu1_lon)) @ Ru_cell @ np.linalg.inv(Cdu1_lon) + Rdu_cell

        H_QP_du_e_lon[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        H_QP_du_e_lon[1, 0] = np.zeros([1, self.Nc * self.Nu])
        H_QP_du_e_lon[1, 1] = self.rou_lon * np.eye(1)
        H_QP_du_e_lon = np.vstack([np.hstack(Mat_size) for Mat_size in H_QP_du_e_lon])
        H_QP_du_e_lon = 2 * H_QP_du_e_lon

        # q为列向量
        f_QP_du_e_lon = np.empty((1, 2), dtype=object)
        f_QP_du_e_lon[0, 0] = 2 * np.transpose(Cy_ext @ (A_ext_lon @ x_current_lon - B_ext_lon @ np.linalg.inv(
            Cdu1_lon) @ Cdu2_lon + C_ext_lon) - self.y_ref_ext_lon) @ Q_cell @ Cy_ext @ B_ext_lon @ np.linalg.inv(
            Cdu1_lon) + 2 * np.transpose(np.linalg.inv(Cdu1_lon) @ (-Cdu2_lon)) @ Ru_cell @ np.linalg.inv(Cdu1_lon)
        f_QP_du_e_lon[0, 1] = np.zeros([1, 1])
        f_QP_du_e_lon = np.vstack([np.hstack(Mat_size) for Mat_size in f_QP_du_e_lon])
        f_QP_du_e_lon = np.transpose(f_QP_du_e_lon)

        lb = np.vstack((self.du_min_ext_lon, self.e_min_lon))
        ub = np.vstack((self.du_max_ext_lon, self.e_max_lon))

        A_du_eCons_lon = np.empty([6, 2], dtype=object)
        ubA_du_eCons_lon = np.empty([6, 1], dtype=object)
        A_du_eCons_lon[0, 0] = np.linalg.inv(Cdu1_lon)
        A_du_eCons_lon[0, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons_lon[1, 0] = B_ext_lon @ np.linalg.inv(Cdu1_lon)
        A_du_eCons_lon[1, 1] = np.zeros([self.Nx * self.Np, 1])
        A_du_eCons_lon[2, 0] = Cy_ext @ B_ext_lon @ np.linalg.inv(Cdu1_lon)
        A_du_eCons_lon[2, 1] = np.zeros([self.Np * self.Ny, 1])
        A_du_eCons_lon[3, 0] = -np.linalg.inv(Cdu1_lon)
        A_du_eCons_lon[3, 1] = np.zeros([self.Nc * self.Nu, 1])
        A_du_eCons_lon[4, 0] = -B_ext_lon @ np.linalg.inv(Cdu1_lon)
        A_du_eCons_lon[4, 1] = np.zeros([self.Nx * self.Np, 1])
        A_du_eCons_lon[5, 0] = -Cy_ext @ B_ext_lon @ np.linalg.inv(Cdu1_lon)
        A_du_eCons_lon[5, 1] = np.zeros([self.Np * self.Ny, 1])
        # A_du_eCons_lon[6, 0] = np.eye(self.Nc*self.Nu)
        # A_du_eCons_lon[6, 1] = np.zeros([self.Nc*self.Nu, self.Nc*self.Nu])
        # A_du_eCons_lon[7, 0] = np.zeros([self.Nc*self.Nu, self.Nc*self.Nu])
        # A_du_eCons_lon[7, 1] = np.eye(self.Nc*self.Nu)
        # A_du_eCons_lon[8, 0] = -np.eye(self.Nc*self.Nu)
        # A_du_eCons_lon[8, 1] = np.zeros([self.Nc*self.Nu, self.Nc*self.Nu])
        # A_du_eCons_lon[9, 0] = np.zeros([self.Nc*self.Nu, self.Nc*self.Nu])
        # A_du_eCons_lon[9, 1] = -np.eye(self.Nc*self.Nu)
        A_du_eCons_lon = np.vstack([np.hstack(Mat_size) for Mat_size in A_du_eCons_lon])

        ubA_du_eCons_lon[0, 0] = self.u_max_ext_lon + Cdu1_lon @ Cdu2_lon
        ubA_du_eCons_lon[
            1, 0] = self.x_max_ext_lon - A_ext_lon @ x_current_lon - C_ext_lon - B_ext_lon @ np.linalg.inv(
            Cdu1_lon) @ (-Cdu2_lon)
        ubA_du_eCons_lon[2, 0] = self.y_max_ext_lon - Cy_ext @ (
                A_ext_lon @ x_current_lon + C_ext_lon + B_ext_lon @ np.linalg.inv(Cdu1_lon) @ (-Cdu2_lon))
        ubA_du_eCons_lon[3, 0] = -self.u_min_ext_lon - np.linalg.inv(Cdu1_lon) @ Cdu2_lon
        ubA_du_eCons_lon[
            4, 0] = -self.x_min_ext_lon + A_ext_lon @ x_current_lon + C_ext_lon - B_ext_lon @ np.linalg.inv(
            Cdu1_lon) @ Cdu2_lon
        ubA_du_eCons_lon[5, 0] = -self.y_min_ext_lon + Cy_ext @ (
                A_ext_lon @ x_current_lon + C_ext_lon - B_ext_lon @ np.linalg.inv(Cdu1_lon) @ Cdu2_lon)
        # ubA_du_eCons_lon[6, 0] = self.du_max_ext_lon
        # ubA_du_eCons_lon[7, 0] = self.e_max_lon
        # ubA_du_eCons_lon[8, 0] = -self.du_min_ext_lon
        # ubA_du_eCons_lon[9, 0] = -self.e_min_lon
        ubA_du_eCons_lon = np.vstack([np.hstack(Mat_size) for Mat_size in ubA_du_eCons_lon])

        # cvxopt求解过程
        P = matrix(H_QP_du_e_lon)
        q0 = f_QP_du_e_lon.astype(np.double)
        q = matrix(q0)
        G = matrix(np.vstack((A_du_eCons_lon, np.eye(self.Nc * self.Nu + 1), -np.eye(self.Nc * self.Nu + 1))))
        h = matrix(np.vstack((ubA_du_eCons_lon, ub, -lb)))

        result = solvers.qp(P, q, G, h)  # 1/2x'Px+q'x   Gx<=h  Ax=b 注意使用qp时，每个参数要换成matrix
        X = result['x']  # 'x'为result中的解，'status'表示是否找到最优解。
        Input = np.hstack([np.linalg.inv(Cdu1_lon), np.zeros([self.Nu * self.Nc, 1])]) @ np.array(X) + np.linalg.inv(Cdu1_lon) @ (
            -Cdu2_lon)

        # u_incr_lon = X[0]  # 控制量增量
        # Input = u_lon_last + u_incr_lon

        return Input[0, 0]
