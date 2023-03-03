from parameter_config import MPC_lat_Config
from parameter_config import MPC_lon_Config
from obstacles_info import ObstaclesInfo
from simulation_env import Simulation
from MPC_controller_lat import MPC_controller_lat
from MPC_controller_lon import MPC_controller_lon
from kinematics_model import Kinematics_Model

if __name__ == '__main__':
    MPC_lon_Config = MPC_lon_Config()
    MPC_lat_Config = MPC_lat_Config()
    Simulation(Kinematics_Model, ObstaclesInfo, MPC_controller_lat, MPC_controller_lon, MPC_lat_Config, MPC_lon_Config)
