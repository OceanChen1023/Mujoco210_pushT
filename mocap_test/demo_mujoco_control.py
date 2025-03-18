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
import click
from typing import Optional
import pathlib
import shutil
import math
from multiprocessing.managers import SharedMemoryManager

from utils.mocap_controller import Controller
from utils.mocap_controller import Direction
from utils.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from utils.replay_buffer import ReplayBuffer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from datetime import datetime
from utils.mujoco_env import Mujoco_Env


DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}
    

def main():
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Mujoco_Env(shm_manager=shm_manager,obs_image_resolution=(640,360),
                       video_capture_resolution=(640,360)
                       
                       
                       ) as env:
            #######    load model & viewer set up camera setting     ############
            #env=Mujoco_Env(shm_manager=shm_manager)
            cv2.setNumThreads(1)
            #########   initialize   .zarr     #################
            #file_path = "Demo/Demo1.zarr"
            file_directory = "Demo"
            files = [f for f in os.listdir(file_directory)]
            file_count = len(files)
            print("file_count",file_count)
            # if os.path.exists(file_path): 
            #     root = zarr.open(file_path, mode="a")  # "a" 模式：如果檔案存在則開啟，不存在則創建
            #     data_group = root.require_group("Data")
            #     meta_group = root.require_group("Meta")
            #     #last_index = meta_group["Meta"]["timestep"][-1,]
            #     print(f"✅ Zarr 資料夾已開啟: {file_path}")
            # else:
            file_name =  file_directory+"/"+f"{file_count + 1}"+"/"+f"{file_count + 1}.zarr"
            root = zarr.open(file_name, mode="w")
            print(f"📂 已新建 Zarr 檔案: {file_name}")
            data_group = root.require_group("Data")
            meta_group = root.require_group("Meta")
            data_group.require_dataset("mocap_pos", shape=(0,3),maxshape=(None,3), dtype=np.float32, chunks=True)
            data_group.require_dataset("mocap_quat", shape=(0,4),maxshape=(None,4), dtype=np.float32, chunks=True)
            data_group.require_dataset("joint_pos", shape=(0,6),maxshape=(None,6), dtype=np.float32, chunks=True)
            data_group.require_dataset("joint_vel", shape=(0,6),maxshape=(None,6), dtype=np.float32, chunks=True)
            data_group.require_dataset("timestep", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
            data_group.require_dataset("actual_time", shape=(0,),maxshape=(None,), dtype=np.float32, chunks=True)
            data_group.require_dataset("timeStamp", shape=(0,),maxshape=(None,), dtype=np.float64, chunks=True)
            meta_group.require_dataset("episode_end", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)

            # ros_thread = threading.Thread(target=env.run_ros_node,args=(env.robot.get_last_state,))
            # ros_thread.start()



            ############  Camera Recorder  #############
            cam_names=["front_cam", "wrist_cam"]
            video_writers={}
            frame_size=(360,240)
            fps=30
            for cam_name in cam_names:
                output_path=file_directory+"/"+f"{file_count+1}/"+f"{cam_name}_video_{file_count+1}.avi"
                video_writers[cam_name]=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'XVID'),fps,frame_size,isColor=True)# postion offset  XVID
            
            
            ########### Initial Robot Position ##########
            env.sim.data.set_mocap_pos("mocap", np.array([0.0, 0.6, 1.236]))  #+ np.array([0.3, 0 , -0.4]))
            env.sim.data.set_mocap_quat("mocap",np.array([0.71,0,0.71,0]))
            for _ in range(100):
                env.sim.step()
                env.viewer.render()
            env.sim.data.set_mocap_pos("mocap", np.array([0.2, 0.4, 0.916]))  #+ np.array([0.3, 0 , -0.4]))



            last_timestamp = time.monotonic() #datetime.now().timestamp()
            diff_interval=0.1
            t_start = time.monotonic()
            iter_idx = 0
            command_latency=0
            hz=10
            dt=1/10
            temp_timestep=env.model.opt.timestep  #0.002
            record_flag=False
            last_robot_data=dict()
            while True: 
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt


                # pump obs
                last_robot_data = env.get_obs()
                #print("Last Robot Data:",last_robot_data)

                #handle key presses
                press_events=key_counter.get_press_events()
                for key_stroke in press_events:
                        print("check key press")
                        if key_stroke == KeyCode(char='escape'):
                            # Exit program
                            stop = True
                            print("exit program")
                        elif key_stroke == KeyCode(char='c'):
                            # Start recording
                            env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                            key_counter.clear()
                            record_flag = True
                            print('Recording!')
                        elif key_stroke == Key.backspace:
                            # Delete the most recent recorded episode
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                                record_flag = False
                stage = key_counter[Key.space]

                env.sim.step()
                env.viewer.render()
                env.viewer.add_extra_menu(env.robot.robot_state['ee_pos'])


                current_timestep = int(env.sim.data.time / env.model.opt.timestep)
                actual_time=current_timestep* temp_timestep
                trigger_interval = int(1/(hz*temp_timestep))
                unix_timestamp=time.monotonic()  #datetime.now().timestamp()
                #print("unix_timestamp:",unix_timestamp)

                if record_flag==True :
                    for cam_name in cam_names:
                        cam_id=env.sim.model.camera_name2id(cam_name)
                        env.render_context.render(frame_size[0],frame_size[1],camera_id=cam_id)
                        bgr_array=env.render_context.read_pixels(frame_size[0],frame_size[1],depth=False)[::-1]
                        rgb_image = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
                        cv2.imshow("MuJoCo View", rgb_image)
                        video_writers[cam_name].write(rgb_image)
                    # Record Zarr    
                    #if current_timestep % ((1/hz)/model.opt.timestep) == 0 :  #10hz
                    if unix_timestamp-last_timestamp >= diff_interval:
                            # meta_group["timestep"].append(np.array([t]))
                            data_group["actual_time"].append(np.array([actual_time]))
                            data_group["timestep"].append(np.array([current_timestep]))
                            data_group["timeStamp"].append(np.array([unix_timestamp]))
                            # data_group["mocap_pos"].append(np.array(ee_Position)) #sim.data.get_mocap_pos("mocap")
                            # data_group["mocap_quat"].append(np.array(ee_quaternion))#sim.data.get_mocap_quat("mocap")
                            # data_group["joint_pos"].append(np.array(joint_pos)) # root["ctrl"][t] = sim.data.ctrl
                            # data_group["joint_vel"].append(np.array(joint_vel)) 
                            last_timestamp=unix_timestamp
                
                if env.viewer.is_running() == False:
                    for writer in video_writers.values():
                        writer.release()  
                    meta_group["episode_end"].append(np.array([current_timestep]))
                    # rclpy.shutdown()
                    # ros_thread.join()
                    break;



if __name__ == '__main__':
    main()
    
