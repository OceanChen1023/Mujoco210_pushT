import zarr
import numpy as np
import pandas as pd

# 打開 Zarr 檔案
#zarr_directory = "Demo"#"/home/ocean/Desktop/Diffusionpolicy/diffusion_policy/data/pusht_real/pusht_real/real_pusht_20230105robot_data/replay_buffer.zarr"



def main():
    demo_count ="replay_buffer"   
    zarr_directory = f"Demo"  #/home/ocean/Desktop/Diffusionpolicy/diffusion_policy/data/pusht_real/pusht_real/real_pusht_20230105robot_data/replay_buffer.zarr"
    zarr_path =  f"{zarr_directory}/{demo_count}.zarr"
    print("zarr_path:",zarr_path)
    store = zarr.open(zarr_path, mode="r")
    # 查看 Zarr 檔案的組織結構
    print("📂 Zarr 檔案結構：")
    print(store.tree())  # 顯示 Group 和 Dataset 結構

    # 讀取數據
    # action = store["data"]["action"][:]
    # robot_eef_pose = store["data"]["robot_eef_pose"][:]  # (100, 50, 7)
    timestep_mark = store["meta"]["episode_ends"][:]
    timestep = store["data"]["timestep"][:]
    robot_action = store["data"]["action"][:] #
    #unix_timestamp = store["data"]["timeStamp"][:]
    robot_eef_pose = store["data"]["mocap_pos"][:]  # (100, 50, 7)
    robot_eef_quat = store["data"]["mocap_quat"][:]  # (100, 50, 7)
    robot_joint_pose = store["data"]["joint_pos"][:]
    robot_joint_vel = store["data"]["joint_vel"][:] # (100
    actual_time = store["data"]["actual_time"][:]
    # img_save_flag = store["Data"]["IMG_Save_Flag"][:]

    print("shape of robot_eef_pose:",robot_eef_pose.shape)
    # robot_eef_pose_vel = store["data"]["robot_eef_pose_vel"][:]
    # robot_joint = store["data"]["robot_joint"][:]  # (100, 50, 7)
    # robot_joint_vel = store["data"]["robot_joint_vel"][:]  # (100, 50, 7)
    # stage = store["data"]["stage"][:]
    # timestamp = store["data"]["timestamp"][:]

    # episode_ends= store["meta"]["episode_ends"][:]



    # 顯示部分數據
    #print("\n🔹 位置數據 (positions) 的形狀：", positions.shape)
    #print("🔹 第一筆操作的第一個時間步：", positions[0, 0])  # 顯示第一筆數據的第一個時間步
    #print("🔹 第一筆操作的時間戳：", timesteps[0, :5])  # 顯示前 5 個時間步的時間戳

    #
    # df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])]) 
    # df_episodes = pd.DataFrame(episode_ends, columns=["episodes_end"])
    # df_timestamp= pd.DataFrame(timestamp,columns=["timestamp"])
    # df_action = pd.DataFrame(action, columns=[f"ac_joint_{i}" for i in range(action.shape[1])])
    # df_robot_eef_pose = pd.DataFrame(robot_eef_pose, columns=[f"ee_joint_{i}" for i in range(robot_eef_pose.shape[1])])
    # df_robot_eef_pose_vel = pd.DataFrame(robot_eef_pose_vel, columns=[f"ee_vel_joint_{i}" for i in range(robot_eef_pose_vel.shape[1])])
    # df_robot_joint = pd.DataFrame(robot_joint, columns=[f"joint_{i}" for i in range(robot_joint.shape[1])])
    # df_robot_joint_vel = pd.DataFrame(robot_joint_vel, columns=[f"vel_joint_{i}" for i in range(robot_joint_vel.shape[1])])
    # df_stage= pd.DataFrame(stage, columns=["stage"])
    # df_final = pd.concat([df_episodes,df_timestamp,df_action,df_robot_eef_pose, df_robot_eef_pose_vel,df_robot_joint,df_robot_joint_vel,df_stage], axis=1)

    df_timestamp_mark= pd.DataFrame(timestep_mark,columns=["episode_ends"])
    df_timestamp= pd.DataFrame(timestep,columns=["timestep"])
    #df_unix_timestamp=pd.DataFrame(unix_timestamp,columns=["unix_timestamp"])
    df_action = pd.DataFrame(robot_action,columns=["action_x","action_y","action_z"])
    df_robot_eef_pose=pd.DataFrame(robot_eef_pose,columns=["eef_pose_x","eef_pose_y","eef_pose_z"])
    df_robot_eef_quat=pd.DataFrame(robot_eef_quat,columns=[f"ee_wxyz{i}" for i in range(robot_eef_quat.shape[1])])
    df_robot_joint_pos=pd.DataFrame(robot_joint_pose,columns=["joint_p1","joint_p2","joint_p3","joint_p4","joint_p5","joint_p6"])
    df_robot_joint_vel=pd.DataFrame(robot_joint_vel,columns=["joint_v1","joint_v2","joint_v3","joint_v4","joint_v5","joint_v6"])
    df_actual_time=pd.DataFrame(actual_time,columns=["actual_time"])
    # dt_img_save_flag=pd.DataFrame(img_save_flag,columns=["Img_save_flag"])
    df_final = pd.concat([df_timestamp_mark,df_timestamp,df_actual_time,df_action,df_robot_eef_pose,df_robot_eef_quat,df_robot_joint_pos,df_robot_joint_vel,df_actual_time],axis=1)
    output_file = f'Demo/{demo_count}.zarr/robot_data.xlsx'
    df_final.to_excel(output_file, index=False, engine="openpyxl")
   

    # # 讀取特定的操作數據 (例如第 10 次操作)
    # operation_id = 10
    # positions_op10 = store["positions"][operation_id]
    # velocities_op10 = store["velocities"][operation_id]

    # print(f"\n✅ 第 {operation_id} 次操作的第一個時間步：", positions_op10[0])

    
if __name__ == '__main__':
    main()