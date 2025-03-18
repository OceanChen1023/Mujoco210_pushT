import gymnasium as gym
from stable_baselines3 import PPO
from pushT_env import PushEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
from stable_baselines3.common.callbacks import BaseCallback





def main():
    class ActionLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(ActionLoggingCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # 取得當前環境的觀察值
            obs = self.training_env.get_attr("state")
            # 取得當前 PPO 預測的動作
            action, _states = self.model.predict(obs, deterministic=False)
            print(f"Step {self.num_timesteps}: Action={action}")
            return True  # 繼續訓練


    # 創建自定義環境
    env = PushEnv("./UR5_pole.xml")
    Vec_env = DummyVecEnv([lambda:env])
    #env.get_wrapper_attr("action")
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 128, 128], vf=[64, 128, 128]))
    

    # 創建 PPO 模型
    model = PPO("MlpPolicy", 
                Vec_env, 
                verbose=1,
                learning_rate=1e-3,
                n_steps=4096,
                batch_size=32,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                tensorboard_log="./tensorboard_logs/",
                policy_kwargs=policy_kwargs)
    
    #get parameters
    #print("parameters:",model.get_parameters())

    # 訓練模型
    model.learn(total_timesteps=10000000)
    
    # 保存模型
    model.save("models/ppo_custom_push_3")
    
    #model=PPO.load("models/ppo_custom_push_reach.zip",env=Vec_env,verbose=1)
    #if trained_model == None:
    #    print("load Model failed")
    # 測試模型
    obs = Vec_env.reset()
    

    for _ in range(1000000):
        action, _states = model.predict(obs)
        print("action:",action)
        obs, reward, collision, info = Vec_env.step(action)
        print("reward=",reward)
        #obs = Vec_env.reset()
        Vec_env.render("human")


if __name__ == "__main__":
    main()
