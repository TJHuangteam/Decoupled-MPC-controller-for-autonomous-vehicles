from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
import imageio, os
import os.path as osp


class Simulation:

    def __init__(self, model, obstacles, lat_controller, lon_controller, lat_param, lon_param):

        self.lat_param = lat_param
        self.lon_param = lon_param
        self.obstacle = obstacles(self.lon_param)
        self.model = model(self.lat_param)
        self.path = 0  # D_ref，可以设置条件令其改变

        self.lat_controller = lat_controller(self.path, self.lat_param, self.model)
        self.lon_controller = lon_controller(self.obstacle.path, self.lon_param, self.model)

        self.steps_N = self.lat_param.N
        Point0 = np.array([0, 0.5, 0])
        self.X_real = np.zeros((self.steps_N, 4))  # X_real = [X , Y , phi , vx]
        self.X_real[0, 0] = self.lon_param.Pos_x_0

        self.X_real[0, 1:3] = Point0[1:3]
        self.X_real[0, 3] = self.lon_param.v_x_0
        self.U_lat_real = np.zeros((self.lon_param.N, 1))  # 存放侧向控制量的矩阵，包含每一步的真实控制量
        self.u_lat_real = np.zeros((self.lat_param.Nu, 1))
        self.Pos_x_obj = np.zeros((self.lon_param.N, 1))
        self.v_x_obj = np.zeros((self.lon_param.N, 1))
        self.U_lon_real = np.zeros((self.lon_param.N, 1))  # 存放纵向控制量的矩阵，包含每一步的真实控制量
        self.u_lon_real = np.zeros((self.lon_param.mpc_Nu, 1))

        self.simulate()
        self.visualization(self.lat_param.sim_result_record_enable)

    def simulate(self):

        start_time = time.time()

        for time_counts in range(self.steps_N - 1):  # 共进行N-1次跟踪，因为每次计算都得到下一个时刻的状态点

            '''
            lat_controller info calculation
            '''
            # 最后一次前面没有参考轨迹了，所以不需要计算到N次
            X_lat = self.X_real[time_counts, 0:3]  # 当前时刻(X,Y,phi)
            X_lon = np.array([self.X_real[time_counts, 0], self.X_real[time_counts, 3]])

            # #给一个换道指令
            if self.X_real[time_counts, 0] >= 1040:
                self.path = self.lat_param.lanewidth

            K = X_lat[2] - 0  # 当frenet坐标系固定在x轴这条路上时
            D = X_lat[1] - 0  # 根据道路信息添加曲率和heading
            x_current_lat = np.array([[K], [D]])

            if time_counts == 0:
                u_lat_last = 0
            else:
                u_lat_last = self.U_lat_real[time_counts - 1, 0]

            vx = self.model.vx
            vy = self.model.vy
            cur = self.model.cur

            '''
            lon_controller info calculation
            '''

            self.Pos_x_obj = self.obstacle.path[time_counts, 1]
            self.v_x_obj = self.obstacle.path[time_counts, 2]

            x_current_lon = np.array([[self.X_real[time_counts, 0]], [self.X_real[time_counts, 3]]])
            u_lon_last = self.U_lon_real[time_counts, 0]

            '''
            lon_controller & lat_controller input calculation
            '''

            Input_lat = self.lat_controller.calc_input(self.path, vx, vy, x_current_lat, u_lat_last, cur)

            Input_lon = self.lon_controller.calc_input(self.Pos_x_obj, self.v_x_obj, x_current_lon,
                                                       u_lon_last, time_counts)

            '''
            lat_controller & lat_controller data record
            '''

            self.u_lat_real = Input_lat
            self.U_lat_real[time_counts + 1, 0] = self.u_lat_real

            self.u_lon_real = Input_lon
            self.U_lon_real[time_counts + 1, 0] = self.u_lon_real

            t = np.arange(0, 0 + 2 * self.lat_param.T, self.lat_param.T)  # 不包含右边界，所以t只有两列

            Point = (X_lat[0], X_lat[1], X_lat[2], X_lon[1])
            print(Input_lon)
            sets = (Input_lat, Input_lon)
            # sets = (0, Input_lon[0, 0])
            # sets = (Input_lat, 0)
            # sets = (0, 0)
            Next_Point = self.model.update(Point, sets, t)  # 第一行为初值，第二行为解

            self.X_real[time_counts + 1, 0] = Next_Point[0]
            self.X_real[time_counts + 1, 1] = Next_Point[1]
            self.X_real[time_counts + 1, 2] = Next_Point[2]
            self.X_real[time_counts + 1, 3] = Next_Point[3]


            # # visualization for debug
            plt.scatter(self.X_real[time_counts, 0], self.X_real[time_counts, 1], marker='o', facecolor='none',
                        edgecolors='b')
            plt.scatter(self.obstacle.path[time_counts, 1], 0, marker='o', facecolor='none',
                        edgecolors='r')
            plt.axis([1000, 1125, -10, 10])
            plt.pause(0.05)


        stop_time = time.time()
        cost = stop_time - start_time
        print("%s Time cost %s second" % (os.path.basename(sys.argv[0]), cost))

    def visualization(self, sim_result_record_enable):
        num = 0
        fig_pos = plt.figure(figsize=(15, 1.8))
        vis_ego = fig_pos.add_subplot(111)

        # simulation_result_record
        if sim_result_record_enable :

            for i in range(0, self.lat_param.N, 5):
                plt.plot([1000, 1130], [0, 0], c='k', ls='--', lw=1, alpha=0.2)
                rect_ego = plt.Rectangle((self.X_real[i, 0] - 4.5, self.X_real[i, 1] - 1.7 / 2), 4.5, 1.7, color='b',
                                         alpha=0.2)
                rect_obj = plt.Rectangle((self.obstacle.path[i, 1], 0 - 1.7 / 2), 4.5, 1.7, color='r',
                                         alpha=0.2)
                # plt.Rectangle给出的是矩形的左下角坐标及长宽信息
                vis_ego.add_patch(rect_ego)
                vis_ego.add_patch(rect_obj)

                plt.scatter(self.X_real[i, 0], self.X_real[i, 1], marker='o', facecolor='none',
                            edgecolors='b')
                plt.scatter(self.obstacle.path[i, 1], 0, marker='o', facecolor='none',
                            edgecolors='r')
                plt.axis([1000, 1130, -4, 4])
                # plt.axis("equal")
                # plt.grid(True)  # 添加网格
                plt.xlabel('Position_X(m)')
                plt.ylabel('Position_Y(m)')
                plt.title('Simulation result')
                plt.savefig('./simulation_result_pic/simulation_result_record/pic-{0:0>4}.png'.format(num + 1))
                num += 1
        else:

            for i in range(0, self.lat_param.N, 5):
                # 绘制simulation_result
                plt.plot([1000, 1130], [0, 0], c='k', ls='--', lw=1, alpha=0.2)
                rect_ego = plt.Rectangle((self.X_real[i, 0] - 4.5, self.X_real[i, 1] - 1.7 / 2), 4.5, 1.7, color='b',
                                         alpha=0.2)
                rect_obj = plt.Rectangle((self.obstacle.path[i, 1], 0 - 1.7 / 2), 4.5, 1.7, color='r',
                                         alpha=0.2)
                # plt.Rectangle给出的是矩形的左下角坐标及长宽信息
                vis_ego.add_patch(rect_ego)
                vis_ego.add_patch(rect_obj)

                plt.scatter(self.X_real[i, 0], self.X_real[i, 1], marker='o', facecolor='none',
                            edgecolors='b')
                plt.scatter(self.obstacle.path[i, 1], 0, marker='o', facecolor='none',
                            edgecolors='r')
                plt.axis([1000, 1130, -4, 4])
                # plt.axis("equal")
                # plt.grid(True)  # 添加网格
                plt.xlabel('Position_X(m)')
                plt.ylabel('Position_Y(m)')
                plt.title('Simulation result')
                plt.savefig('./simulation_result_pic/simulation_result/pic-{0:0>4}.png'.format(num + 1))
                plt.cla()
                num += 1

        if sim_result_record_enable:
            img_dir = './simulation_result_pic/simulation_result_record/'
        else:
            img_dir = './simulation_result_pic/simulation_result/'

        frames = []

        for idx in sorted(os.listdir(img_dir)):
            img = osp.join(img_dir, idx)
            frames.append(imageio.imread(img))
        par_dir = osp.dirname(img_dir)

        if sim_result_record_enable:
            gif_path = osp.join(par_dir, 'simulation_result_record.gif')
        else:
            gif_path = osp.join(par_dir, 'simulation_result.gif')

        imageio.mimsave(gif_path, frames, 'GIF', duration=0.5)
        print('Finish changing!')

        """
        :param img_dir: 包含图片的文件夹
        :param gif_path: 输出的gif的路径
        :param duration: 每张图片切换的时间间隔，与fps的关系：duration = 1 / fps
        :return:
        """
