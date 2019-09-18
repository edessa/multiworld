import multiworld
import gym
import mujoco_py
import cv2
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import *
import time

imsize = 48
multiworld.register_all_envs()
#env = gym.make('SawyerPickupMultiobj-v0')
#env = gym.make('SawyerPickupEnv-v0')
<<<<<<< HEAD
#env = gym.make('SawyerPickupEnv-v1')
env = gym.make('SawyerPushNESW-v0')
#env = gym.make('SawyerPushNIPS-v0')
=======
env = gym.make('SawyerMultiObj-v0')
>>>>>>> b1670b556462cebae3829756ea0e737aa9537619

env = ImageEnv(
    env,
    imsize = imsize,
<<<<<<< HEAD
    init_camera=sawyer_init_camera_zoomed_in,
=======
    init_camera=sawyer_pusher_camera_upright_v1,
>>>>>>> b1670b556462cebae3829756ea0e737aa9537619
    transpose=True,
    normalize=True,

)
i = 0


for j in range(0, 5):
    obs = env.reset()
    print("reset")
    action = np.array([1,1,1,1])

    obs_img = 255*obs['observation'].reshape(3, 48, 48).transpose()
    cv2.imwrite('obs' + str(j) + '.png', obs_img)
    obs_2_img = 255*obs['image_desired_goal'].reshape(3, 48, 48).transpose()
    cv2.imwrite('goal' + str(j) + '.png', obs_2_img)

#goal = env.sample_goal()
#obs_img = 255*goal['desired_goal'].reshape(3, 480, 480).transpose()
#cv2.imwrite('a.png', obs_img)
#cv2.imshow('window', obs_img)
#cv2.waitKey()
