from mujoco_py import MjSim, MjViewer, load_model_from_path
import numpy as np
from scipy.spatial.transform import Rotation as R
import glfw
from mujoco_py import const
from enum import Enum
import cv2
import mujoco_py
import threading
import mujoco
import zarr
import os
import time
import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from datetime import datetime
from utils.pytorch_util import dict_apply


Start_Record = {"o":False,"p":False} #False
#Ros2 Publisher----------------------------------------------------------------
class MinimalPublisher(Node):

    def __init__(self,robotpos_callback):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self._robot_position=robotpos_callback

    def timer_callback(self):
        msg = String()
        rotpos=self._robot_position()
        msg.data = 'Robot position= %s' % rotpos   #'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1



#Main Ctrl----------------------------------------------------------------
def rotation(theta_x=0, theta_y=0, theta_z=0):

    rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = rot_x.dot(rot_y).dot(rot_z)

    return R

def quat2euler(quat):
    # transfer quat to euler
    r = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))
    return r.as_euler('XYZ')



class Direction(Enum):
    POS: int = 1
    NEG: int = -1

class Controller():
    # The max speed.
    MAX_SPEED = 1.0

    # The minimum speed.
    MIN_SPEED = 0.0
    SPEED_CHANGE_PERCENT = 0.2

    def __init__(self, sim) -> None:
        super().__init__()
        self._speeds =  np.array([0.01, 0.1])
        self.sim = sim

    @property
    def pos_speed(self):
        """
        The speed that arm moves.
        """
        return self._speeds[0]

    @property
    def rot_speed(self):
        """
        The speed that wrist rotates.
        """
        return self._speeds[1]

    def speed_up(self):
        """
        Increase gripper moving speed.
        """
        self._speeds = np.minimum(
            self._speeds * (1 + self.SPEED_CHANGE_PERCENT), self.MAX_SPEED
        )
    def speed_down(self):
        """
        Decrease gripper moving speed.
        """
        self._speeds = np.maximum(
            self._speeds * (1 - self.SPEED_CHANGE_PERCENT), self.MIN_SPEED
        )

    def move_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._move(0, direction)

    def move_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._move(1, direction)

    def move_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._move(2, direction)


    def rot_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._rot(0, direction)

    def rot_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._rot(1, direction)

    def rot_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._rot(2, direction)

    def _rot(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        e = quat2euler(self.sim.data.mocap_quat[0])
        if axis == 2:
            r = R.from_matrix(rotation(e[0] , e[1], e[2] + self.rot_speed * direction.value))
            self.sim.data.set_mocap_quat("mocap",np.array([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]]) )
            self.sim.step()
        elif axis == 1:
            r = R.from_matrix(rotation(e[0] , e[1] + self.rot_speed * direction.value, e[2]))
            self.sim.data.set_mocap_quat("mocap",np.array([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]]) )
        elif axis == 0:
            r = R.from_matrix(rotation(e[0] + self.rot_speed * direction.value, e[1], e[2]))
            self.sim.data.set_mocap_quat("mocap",np.array([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]]) )
        else: 
            pass

    def _move(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        if axis == 2:  #Z axis
            self.sim.data.set_mocap_pos("mocap", self.sim.data.mocap_pos +  np.array([0, 0, self.pos_speed * direction.value]))
            self.sim.step()
        elif axis == 0: #y axis
            self.sim.data.set_mocap_pos("mocap", self.sim.data.mocap_pos +  np.array([0, self.pos_speed * direction.value, 0]))
            self.sim.step()
        elif axis == 1: #x axis
            self.sim.data.set_mocap_pos("mocap", self.sim.data.mocap_pos +  np.array([self.pos_speed * direction.value, 0, 0]))
            self.sim.step()
        else: 
            pass

    def record_Start(self,boolean:bool):
        if boolean:
            Start_Record["o"]= True
            print("Record Flag=1")
        else:
            Start_Record["o"]= False         
            Start_Record["p"]= False
            print("Record Flag=0")

class Viewer(MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.controller =  Controller(sim)
        self.running = True
        self.reset = False
        self.record_flag = False
    def key_callback(self, window, key, scancode, action, mods):
        # Trigger on keyup only:
        if key == glfw.KEY_UP:
            self.controller.move_z(Direction.POS)

        elif key == glfw.KEY_DOWN:
            self.controller.move_z(Direction.NEG)

        elif key == glfw.KEY_RIGHT:
            self.controller.move_y(Direction.POS)

        elif key == glfw.KEY_LEFT:
            self.controller.move_y(Direction.NEG)

        elif key == glfw.KEY_M:
            self.controller.move_x(Direction.NEG) 

        elif key == glfw.KEY_J:
            self.controller.move_x(Direction.POS)

        elif key == glfw.KEY_A:
            self.controller.rot_y(Direction.POS)

        # elif key == glfw.KEY_S:
        #     self.controller.rot_y(Direction.NEG)

        elif key == glfw.KEY_Q:
            self.controller.rot_x(Direction.POS)

        elif key == glfw.KEY_W:
            self.controller.rot_x(Direction.NEG)

        elif key == glfw.KEY_Z:
            self.controller.rot_z(Direction.POS)

        elif key == glfw.KEY_X:
            self.controller.rot_z(Direction.NEG)

        elif key == glfw.KEY_MINUS:
            self.controller.speed_down()

        elif key == glfw.KEY_EQUAL:
            self.controller.speed_up()
        elif key == glfw.KEY_ESCAPE:
            self.running = False
        elif key == glfw.KEY_O:
            self.record_flag = True
        else:
            super().key_callback(window, key, scancode, action, mods)


        # if action == glfw.PRESS:
        #     if key  == glfw.KEY_O:
        #         if Start_Record["o"]==True:
        #             print("Stop Recording")
        #             Start_Record["o"] = False
        #         elif Start_Record["o"]==False:
        #             print("Start Recording")
        #             Start_Record["o"] = True
                    
    def is_running(self):
        return self.running
                        
    def resetflag(self):
        return self.resetdata_group
    
    def resetflag_reset(self):
        self.reset=False

    def record_flag(self):
        return self.record_flag


    def add_extra_menu(self,robot_position):
        self.add_overlay(
            const.GRID_TOPRIGHT,
            "Go up/down/left/right",
            "[up]/[down]/[left]/[right] arrow",
        )
        self.add_overlay(const.GRID_TOPRIGHT, "Go forwarf/backward", "[J]/[M]")
        self.add_overlay(const.GRID_TOPRIGHT, "ROT_X", "[Q]/[W]")
        self.add_overlay(const.GRID_TOPRIGHT, "ROT_Y", "[A]/[S]")
        self.add_overlay(const.GRID_TOPRIGHT, "ROT_Z", "[Z]/[X]")
        self.add_overlay(const.GRID_TOPRIGHT, "Slow down/Speed up", "[-]/[=]")
        self.add_overlay(const.GRID_TOPRIGHT, "Start/Stop Record", "[O]")
        self.add_overlay(const.GRID_TOPRIGHT, "RobotPosition:",str(robot_position))
        



def run_ros_node(get_robot_position,args=None):
    """Run the ROS 2 node in a separate thread."""
    rclpy.init(args=args)
    minimal_publisher=MinimalPublisher(get_robot_position)
    try:
        rclpy.spin(minimal_publisher)
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()



def main():

    #######    load model & viewer set up camera setting     ############
    model = load_model_from_path("./UR5_pole.xml")
    sim = MjSim(model)
    viewer = Viewer(sim)
    body_id = sim.model.body_name2id('Pole')
    lookat = sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 4
    viewer.cam.azimuth = -90.
    viewer.cam.elevation = -15
    render_context=mujoco_py.MjRenderContextOffscreen(sim,None,1)
    render_context.vopt.geomgroup[2]=1
    render_context.vopt.geomgroup[1]=1
    sim.add_render_context(render_context)

    #########   initialize   .zarr     #################
    #file_path = "Demo/Demo1.zarr"
    file_directory = "Demo"
    files = [f for f in os.listdir(file_directory) if f.endswith(".zarr")]
    file_count = len(files)
    print("file_count",file_count)
    # if os.path.exists(file_path): 
    #     root = zarr.open(file_path, mode="a")  # "a" 模式：如果檔案存在則開啟，不存在則創建
    #     data_group = root.require_group("Data")
    #     meta_group = root.require_group("Meta")
    #     #last_index = meta_group["Meta"]["timestep"][-1,]
    #     print(f"✅ Zarr 資料夾已開啟: {file_path}")
    # else:
    file_name =  file_directory+"/"+f"{file_count}.zarr"
    root = zarr.open(file_name, mode="w")
    print(f"📂 已新建 Zarr 檔案: {file_name}")
    data_group = root.require_group("data")
    meta_group = root.require_group("meta")
    data_group.require_dataset("action", shape=(0,3),maxshape=(None,3), dtype=np.float32, chunks=True)
    data_group.require_dataset("mocap_pos", shape=(0,3),maxshape=(None,3), dtype=np.float32, chunks=True)
    data_group.require_dataset("mocap_quat", shape=(0,4),maxshape=(None,4), dtype=np.float32, chunks=True)
    data_group.require_dataset("joint_pos", shape=(0,6),maxshape=(None,6), dtype=np.float32, chunks=True)
    data_group.require_dataset("joint_vel", shape=(0,6),maxshape=(None,6), dtype=np.float32, chunks=True)
    data_group.require_dataset("timestep", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
    data_group.require_dataset("actual_time", shape=(0,),maxshape=(None,), dtype=np.float32, chunks=True)
    #data_group.require_dataset("timeStamp", shape=(0,),maxshape=(None,), dtype=np.float64, chunks=True)
    # data_group.require_dataset("IMG_Save_Flag", shape=(0,),maxshape=(None,),dtype=np.int32, chunks=True)
    meta_group.require_dataset("episode_ends", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
   
   

    def get_robot_position():
        return sim.data.get_mocap_pos("mocap")

    def _get_obs():
        eef_pos=sim.data.get_mocap("mocap")
        for i in enumerate(cam_names):
            render_context.render(frame_size[0],frame_size[1],camera_id=i)
            bgr_array=render_context.read_pixels(frame_size[0],frame_size[1],depth=False)[::-1]
            rgb_image[i] = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        
        return eef_pos,rgb_image

    ros_thread = threading.Thread(target=run_ros_node,args=(get_robot_position,))
    ros_thread.start()
    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)  # Create a window to init GLFW.


    ############  Camera Recorder  #############
    cam_names=["front_cam", "wrist_cam"]
    video_writers={}
    frame_size=(1280,720)
    fps=30
    for i,cam_name in enumerate(cam_names):
        output_path=file_directory+"/"+"videos"+"/"+str(file_count)
        os.makedirs(output_path, exist_ok=True)
        output_path=output_path+"/"+f"{i}.mp4"
        video_writers[cam_name]=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'XVID'),fps,frame_size,isColor=True)# postion offset  XVID
    
    
    
    sim.data.set_mocap_pos("mocap", np.array([0.0, 0.6, 1.236]))  #+ np.array([0.3, 0 , -0.4]))
    sim.data.set_mocap_quat("mocap",np.array([0.71,0,0.71,0]))
    for _ in range(100):
        sim.step()
        viewer.render()
    sim.data.set_mocap_pos("mocap", np.array([0.2, 0.4, 0.916]))  #+ np.array([0.3, 0 , -0.4]))
    
    #sim.data.qpos[0:6]=np.array([0.113,-0.163,1.79,-0.046,-0.26,-0])
    #T_block_site_id = sim.model.site_name2id("T_block_site")
    #print("T_shape_site_id",T_block_site_id)
    #sim.model.site_rgba[T_block_site_id]=[0., 0., 1., 1.]

    device = torch.device('cuda')
    MODEL_PATH = "checkpoints/0318.ckpt"  # 替換成你的模型檔案路徑
    # evl_model = torch.load(MODEL_PATH)
    # evl_model.eval().to(device)
    evl_model = torch.jit.load(MODEL_PATH, map_location="cpu")
    evl_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    #policy.eval().to(device)    
    last_timestamp = time.monotonic() #datetime.now().timestamp()
    diff_interval=0.1
    hz=10
    counter=50
    interval=50/3
    step_counter=0
    ## action is absolute position
    robot_Position=sim.data.get_mocap_pos("mocap")

    while True: 

                #T_block_position = sim.data.site_xpos[T_block_site_id]  # 获取对应的全局位置    

        for joint_id in range(sim.model.njnt):
            joint_name = sim.model.joint_id2name(joint_id)
            joint_dofadr = sim.model.jnt_dofadr[joint_id]  # DOF 索引
            dof_position = sim.data.qpos[joint_dofadr]
            dof_velocity = sim.data.qvel[joint_dofadr]

            #print(f"Joint Name: {joint_name} Position (qpos): {dof_position}")
            #print(f"  Velocity (qvel): {dof_velocity}")
        joint_pos=sim.data.qpos[0:6].reshape(1,6)
        joint_vel=sim.data.qvel[0:6].reshape(1,6)

      
        mocap_quaternion = sim.data.get_mocap_quat("mocap")
        mocap_quaternion=mocap_quaternion.reshape(1,4)
 
        action=sim.data.get_mocap_pos("mocap")






        sim.step()
        viewer.render()
        #robot_Position=robot_Position
        
        viewer.add_extra_menu(action)
        # 依 timestep 存數據
        temp_timestep=model.opt.timestep  #0.002
        # print("temp_timestep:",temp_timestep)
        current_timestep = int(sim.data.time / model.opt.timestep)
        #print("sim.data.time:",sim.data.time)
        #print("current_timestep:",current_timestep)
        actual_time=current_timestep* temp_timestep
        # print("actual_time:",actual_time)
        trigger_interval = int(1/(hz*temp_timestep))
        #print("trigger_interval:",trigger_interval)
        
        #unix_timestamp=time.monotonic()  #datetime.now().timestamp()
        #print("unix_timestamp:",unix_timestamp)




        if viewer.record_flag==True :
             # Record video
            #print("Start Record Time:",datetime.now())
            if counter % (interval) < 1:
                print("Video writer write img")
                    # get obs
                print('get_obs')
                obs = _get_obs()
                obs_timestamps = obs['timestamp']
                print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                # run inference
                with torch.no_grad():
                    s = time.time()
                    obs_dict_np = obs  #_get_obs()
                    print("obs_dict_np:",obs_dict_np)
                    obs_dict = dict_apply(obs_dict_np, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = evl_model.predict_action(obs_dict)
                    # this action starts from the first obs step
                    action = result['action'][0].detach().to('cpu').numpy()
                    print('Inference latency:', time.time() - s)
                for cam_name in cam_names:
                    cam_id=sim.model.camera_name2id(cam_name)
                    #print("Cam_id:",cam_id)
                    render_context.render(frame_size[0],frame_size[1],camera_id=cam_id)
                    bgr_array=render_context.read_pixels(frame_size[0],frame_size[1],depth=False)[::-1]
                    rgb_image = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
                    video_writers[cam_name].write(rgb_image)

                
                 # Record Zarr    
                if counter % (50) == 0 :  #10hz
                #if unix_timestamp-last_timestamp >= diff_interval:
                        #meta_group["timestep"].append(np.array([t]))
                        data_group["actual_time"].append(np.array([actual_time]))
                        data_group["action"].append(np.array(action.reshape(1,3)))
                        data_group["timestep"].append(np.array([current_timestep]))
                        data_group["mocap_pos"].append(np.array(robot_Position.reshape(1,3))) #sim.data.get_mocap_pos("mocap")
                        data_group["mocap_quat"].append(mocap_quaternion)#sim.data.get_mocap_quat("mocap")
                        data_group["joint_pos"].append(np.array(joint_pos)) # root["ctrl"][t] = sim.data.ctrl
                        data_group["joint_vel"].append(np.array(joint_vel)) 
                        robot_Position=action.copy()
        
                    
                        counter=50
                        step_counter+=1

        if viewer.is_running() == False:
            print("step_counter:",step_counter)
            for writer in video_writers.values():
                writer.release()  
            meta_group["episode_ends"].append(np.array([step_counter]))
            rclpy.shutdown()
            ros_thread.join()
            break;

        counter-=1


if __name__ == '__main__':
    main()
    
