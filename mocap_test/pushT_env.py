import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import glfw

        

class PushEnv(gym.Env):
   
    total_reward=0
    def __init__(self,model_xml_path):
        super().__init__()
        # Load MuJoCo model
        self.model = load_model_from_path(model_xml_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        render=True

        # 是否顯示 MuJoCo Viewer
        self.render_enabled = render
        if self.render_enabled:
            glfw.init()  # 確保 glfw 有被初始化
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = None
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)   #observation space={ee_x,ee_y,ee_z}

        #self.sim.forward()
        #self.ee_init_pos =self._get_ee_pos() #np.array(3, dtype=np.float32)
        self._set_robot_pose()
        self.block_init_pos = self._get_block_init_pos() #np.array(3,dtype=np.float32)
        #print("ee init pos: ",self.ee_init_pos)
        self.phase=0 
        self.pre_pos=np.array([0,0,0],dtype=np.float32)
        self.cur_pos=np.array([0,0,0],dtype=np.float32)
        self.temp_ee_pos_clip=np.array([0,0,0], dtype=np.float32)

    def reset(self):
        self.sim.reset()
        self.phase=0
        # 重置末端位置
        #init_pos = np.array([0.5, 0.2, 1.8])
        #block_init_pos=np.array([0.6, 0.0, 1.1])
        super().__init__()
        #self._set_ee_pos(self.ee_init_pos)
        self._set_robot_pose()
        self._set_block_pos(self.block_init_pos)
        print("reset")
        if self.render_enabled and self.viewer is not None:
            self.render()
        return self._get_obs()

    def _set_robot_pose(self):
        print("set robot pose")
        self.sim.data.set_mocap_pos("mocap", np.array([0.2, 0.0, 1.236]))  #+ np.array([0.3, 0 , -0.4]))
        self.sim.data.set_mocap_quat("mocap",np.array([0.71,0,0.71,0]))
        #target_positions = [0.00, -0.16, 1.794, 0.0, -5.008, 0.0234]
        #for i, joint_name in enumerate(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]):
        #    joint_id = self.sim.model.joint_name2id(joint_name)  # 獲取關節 ID
        #    dof_index = self.sim.model.jnt_qposadr[joint_id]    # 獲取 DOF 在 qpos 中的索引

        #    self.sim.data.qpos[dof_index] = target_positions[i]
        # # 設置目標位置
        self.sim.forward()
        #return self.sim.data.qpos[dof_index] 
    
    def _get_robot_pose(self):
        for i, joint_name in enumerate(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]):
            joint_id = self.sim.model.joint_name2id(joint_name)  # 獲取關節 ID
            dof_index = self.sim.model.jnt_qposadr[joint_id]    # 獲取 DOF 在 qpos 中的索引

            print("qpos=",self.sim.data.qpos[dof_index])
            
    def _get_ee_pos(self):
        return self.sim.data.get_mocap_pos("mocap")

    def _get_ee_quat(self):
        return self.sim.data.get_mocap_quat("mocap")
    
    def _set_ee_pos(self, new_pos):
        self.sim.data.set_mocap_pos("mocap", new_pos)
        self.sim.forward()
    
    def _get_ee_init_pos(self):
        ee_id=self.model.body_name2id("Pole")
        return self.model.body_pos[ee_id]
  
    def _get_block_pos(self):
        temp_b_ID = self.sim.model.body_name2id("T_block")
        pos=self.sim.data.body_xpos[temp_b_ID]
        #print("block_pos=",pos)
        return pos

    def _set_block_pos(self,new_pos):
        temp_b_ID = self.sim.model.body_name2id("T_block")
        self.sim.data.body_xpos[temp_b_ID] = new_pos
        self.sim.forward()
       

    def _get_block_init_pos(self):
        b_id=self.model.body_name2id("T_block")
        print("block_init_pos=",self.model.body_pos[b_id])
        return self.model.body_pos[b_id]
    

    def render(self, mode="human"):
        self.viewer.render()

    def _set_ee_pos(self, new_pos):
        """設定末端 (mocap) 的位置"""
        self.sim.data.set_mocap_pos("mocap", new_pos)
        self.sim.forward()    


    def _get_obs(self):
        # Return observation (robot state + object position + goal position)
        ee_pos = self._get_ee_pos()
        block_pos = self._get_block_pos()
        return np.concatenate([
            ee_pos,
            block_pos,
            self.sim.data.qpos,  # 關節角度
            self.sim.data.qvel,  # 關節速度
            #self.sim.data.body_xpos[self.ee_id],  # 末端效應器位置
        ]).astype(np.float32)
    
        
    def _calculate_reward(self,obs,pre_pos,cur_pos):

        
        #(phase 0:reaching, phase 1 :Push)
        # Calculate distance between object and goal
        reward=0
        quat_penalty=0
        collision = False
        object_pos = cur_pos
        mocap_qw,mocap_qx,mocap_qy,mocap_qz = self._get_ee_quat()  #[0.71,0,0.71,0]
        #print("mocap_quat",mocap_qw,mocap_qx,mocap_qy,mocap_qz)
        
        #self._get_robot_pose()
        #block_init_pos=np.array([1, -0.2, 1.2])
        goal_pos = np.array([0.2, 0.0, 1.5]) #self.block_init_pos
        #print("goal_pos",goal_pos)
        #print("object_pos",object_pos)
        dist = np.linalg.norm(object_pos - goal_pos)
        reach_reward = 1-np.tanh(2*dist)
        #print("reach_reward",reach_reward)  
        #pose_penalty = -np.sum(np.abs(obs[6:11])) 
        #print("pose_penalty",pose_penalty)
        dw = np.abs(mocap_qw - 0.71)/0.4
        dy = np.abs(mocap_qy - 0.71)/0.4
        quat_penalty=(1-np.tanh(5*dw)+1-np.tanh(5*dy))/2
        print("cur_pos:",cur_pos,"  Present object,",pre_pos)
        step_dist=np.linalg.norm(cur_pos-pre_pos)
        print("step_dist:",step_dist)
        if self.phase==0:
          
            reward=0.2*reach_reward #+0.6*quat_penalty #+pose_penalty 
            print("step_dist:",step_dist, " distance:",dist," reach_reward",reach_reward)
            if step_dist<0.01:
                reward-=1
                print("step_dist<0.01:")
            if dist >1:
                reward-=0.5
            if dist < 0.1:
              reward+=10
              print("reward+10")
              #self.phase=1
              # Success condition
            #print("block_init_pos",self.block_init_pos,"block_cur_pos",self._get_block_pos())
            #flag=np.allclose(self._get_block_pos(), self.block_init_pos, atol=1e-3)
            #print("flag",flag)
            #print("block_cur_pos=",self._get_block_pos(),"block_init_pose=",self.block_init_pos)
            if not np.allclose(self._get_block_pos(), self.block_init_pos,atol=1e-1):
                #print("phase=0")
                collision = True
                #print("Block position",self._get_block_pos(),"Block_init",self.block_init_pos)
                if collision == True:    
                    reward=-3
                    self.phase=0
        total_reward=+reward

        #print("total_reward",total_reward)
        return reward, collision

    def step(self, action):

        temp_ee_pos=self._get_ee_pos()
        self.pre_pos=temp_ee_pos.copy()
        temp_ee_pos+=action
        #print("action: " , action)
        # Apply action
        #self.sim.data.set_mocap_pos[:] = temp_ee_pos
        
        temp_ee_pos[0] = np.clip(temp_ee_pos[0],-0.1, 1)
        temp_ee_pos[1] = np.clip(temp_ee_pos[1], -0.8, 0.8)
        temp_ee_pos[2] = np.clip(temp_ee_pos[2], 1.4, 1.4)
        self.sim.data.set_mocap_pos("mocap", temp_ee_pos)
        self.sim.forward()
        
        if self.render_enabled and self.viewer is not None:
            self.render()


        for _ in range(5):  # sub-steps
            self.sim.step()
        self.cur_pos=temp_ee_pos
             
        obs=self._get_obs()
        #print("obs: ",obs[0:6])
        #print("present_pos: ",self.pre_pos)
        #print("current_pos: ",self.cur_pos)
        reward, collision = self._calculate_reward(obs,self.pre_pos,self.cur_pos)
        if collision==True:
            print("collision==True")
            reward= -5
            self.reset()
        return self._get_obs(), reward, collision, {}