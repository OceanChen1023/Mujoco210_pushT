from typing import Optional
import pathlib
import numpy as np
import time
import glfw
import shutil
import math
from enum import Enum
import cv2
import mujoco_py
import threading
import mujoco
from mujoco_py import const
import zarr
import os
import time
import click
from typing import Optional
import pathlib
import shutil
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from mujoco_py import MjSim, MjViewer, load_model_from_path

from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
#from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
#from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from utils.video_recorder import VideoRecorder
from utils.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
#from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from utils.replay_buffer import ReplayBuffer
from utils.cv2_util import (
    get_image_transform, optimal_row_cols)

from utils.mocap_controller import Controller
from utils.mocap_controller import Direction
from utils.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from utils.replay_buffer import ReplayBuffer
from utils.mujoco_multi_cam import Mujoco_Multi_Cam

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


class Viewer(MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.controller =  Controller(sim)
        self.direction = Direction
        self.running = True
        self.reset = False
    def key_callback(self, window, key, scancode, action, mods):
        # Trigger on keyup only:
        if key == glfw.KEY_UP:
            self.controller.move_z(self.direction.POS)

        elif key == glfw.KEY_DOWN:
            self.controller.move_z(self.direction.NEG)

        elif key == glfw.KEY_RIGHT:
            self.controller.move_y(self.direction.POS)

        elif key == glfw.KEY_LEFT:
            self.controller.move_y(self.direction.NEG)

        elif key == glfw.KEY_M:
            self.controller.move_x(self.direction.NEG) 

        elif key == glfw.KEY_J:
            self.controller.move_x(self.direction.POS)

        elif key == glfw.KEY_A:
            self.controller.rot_y(self.direction.POS)

        # elif key == glfw.KEY_S:
        #     self.controller.rot_y(Direction.NEG)

        elif key == glfw.KEY_Q:
            self.controller.rot_x(self.direction.POS)

        elif key == glfw.KEY_W:
            self.controller.rot_x(self.direction.NEG)

        elif key == glfw.KEY_Z:
            self.controller.rot_z(self.direction.POS)

        elif key == glfw.KEY_X:
            self.controller.rot_z(self.direction.NEG)

        elif key == glfw.KEY_MINUS:
            self.controller.speed_down()

        elif key == glfw.KEY_EQUAL:
            self.controller.speed_up()
        elif key == glfw.KEY_ESCAPE:
            self.running = False
        else:
            super().key_callback(window, key, scancode, action, mods)


                    
    def is_running(self):
        return self.running
                        
    def resetflag(self):
        return self.reset
    
    def resetflag_reset(self):
        self.reset=False



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
        self.add_overlay(const.GRID_TOPRIGHT, "Start Record", "[c]")
        self.add_overlay(const.GRID_TOPRIGHT, "RobotPosition:",str(robot_position))
        
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

class Mujoco_Robot():
    def __init__(self,sim):
        self.sim=sim
        self.robot_state = {
            'joint_pos' : np.zeros((1,6)), 
            'joint_vel' : np.zeros((1,6)),
            'ee_pos' : np.zeros((1,3)),
            'ee_quat' : np.zeros((1,4)),
            'robot_receive_timestamp' : 0
        }


    def get_last_state(self):
        self.robot_state['joint_pos'] = self.sim.data.qpos[0:6].reshape(1,6)  
        self.robot_state['joint_vel'] = self.sim.data.qvel[0:6].reshape(1,6)
        self.robot_state['ee_pos'] = self.sim.data.get_mocap_pos("mocap").reshape(1,3)
        self.robot_state['ee_quat'] = self.sim.data.get_mocap_quat("mocap").reshape(1,4)
        self.robot_state['robot_receive_timestamp'] = time.monotonic()
        return self.robot_state


class Mujoco_Env:    
    def __init__(self,
                 # required params
            output_dir="Demo/",
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            obs_image_resolution=None,
            max_obs_buffer_size=30,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=None,
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(360,240),
            # shared memory
            shm_manager=None):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        
        self.initial_flag = False
        self.model = load_model_from_path("./UR5_pole.xml")
        self.sim = MjSim(self.model)
        self.viewer = Viewer(self.sim)
        self.body_id = self.sim.model.body_name2id('Pole')

        self.robot= Mujoco_Robot(self.sim)
        camera_names = ["front_cam","wrist_cam"]
        self.lookat = self.sim.data.body_xpos[self.body_id]
        for idx, value in enumerate(self.lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = 4
        self.viewer.cam.azimuth = -90.
        self.viewer.cam.elevation = -15
        self.render_context=mujoco_py.MjRenderContextOffscreen(self.sim,None,1)
        self.render_context.vopt.geomgroup[2]=1
        self.render_context.vopt.geomgroup[1]=1
        self.sim.add_render_context(self.render_context)



        self.cameras = Mujoco_Multi_Cam(mujoco_sim=self.sim,cam_names=camera_names,render_context_env=self.render_context,resolution=obs_image_resolution)

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

    
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        # self.max_rot_speed = max_rot_speedrealsense
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        # self.last_realsense_data = None
        self.mujoco_last_data = None
        self.mujoco_cam_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf

        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255
        def transform(data):
            data['color'] = color_transform(data['color'])
            return data
        
        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)
        # self.robot.stop_wait()
        # self.realsense.stop_wait()
        # if self.multi_cam_vis is not None:
        #     self.multi_cam_vis.stop_wait()

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.cameras.is_ready #and self.robot.is_ready
    
    def start(self, wait=True):
        self.cameras.start(wait=False)
        #self.robot.start(wait=False)
        # if self.multi_cam_vis is not None:
        #     self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        # if self.multi_cam_vis is not None:
        #     self.multi_cam_vis.stop(wait=False)
        #self.robot.stop(wait=False)
        self.cameras.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.cameras.start_wait()
        #self.robot.start_wait()
        # if self.multi_cam_vis is not None:
        #     self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        #self.robot.stop_wait()
        self.cameras.stop_wait()
        # if self.multi_cam_vis is not None:
        #     self.multi_cam_vis.stop_wait()



    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


    # =========  ROS  &   API     ===========   
    def run_ros_node(self,lamba_robot_position,args=None):
        """Run the ROS 2 node in a separate thread."""
        rclpy.init(args=args)
        minimal_publisher=MinimalPublisher(lamba_robot_position)
        try:
            rclpy.spin(minimal_publisher)
        finally:
            minimal_publisher.destroy_node()
            rclpy.shutdown()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        #assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        # print("k: ", k)
        self.mujoco_cam_data = self.cameras.get(
            k=k, 
            out=self.mujoco_cam_data)


        self.render_context.render(1280,720,camera_id=0)
        bgr_array=self.render_context.read_pixels(1280,720,depth=False)[::-1]
        #rgb_image = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        bgr_array = np.clip(bgr_array, 0, 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)

        cv2.imshow("MuJoCo Camera View", rgb_image)



        # 125 hz, robot_receive_timestamp , Replace with Mujoco Robot Env
        last_robot_data = self.robot.get_last_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.mujoco_cam_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.mujoco_cam_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        if isinstance(robot_timestamps, float):
            robot_timestamps = [robot_timestamps]  # 轉換為單元素列表
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
        
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )


    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.cameras.n_cameras  #realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.cameras.restart_put(start_time=start_time)
        self.cameras.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        #self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

