import multiworld
import gym
import mujoco_py
import cv2
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import *
import time

imsize = 480
multiworld.register_all_envs()
#env = gym.make('SawyerPickupMultiobj-v0')
env = gym.make('SawyerPickupWideEnv-v0')
#env = gym.make('SawyerMultiObj-v0')
#env = gym.make('SawyerPushNIPS-v0'

env = ImageEnv(
    env,
    imsize = imsize,
    init_camera=init_sawyer_camera_v5,
    transpose=True,
    normalize=True,

)
i = 0


for j in range(0, 20):
    obs = env.reset()
    print("reset")
    action = np.array([1,1,1,1])

    obs_img = 255*obs['observation'].reshape(3, 480, 480).transpose()
    cv2.imwrite('obs' + str(j) + '.png', obs_img[...,::-1])
    obs_2_img = 255*obs['image_desired_goal'].reshape(3, 480, 480).transpose()
    cv2.imwrite('goal' + str(j) + '.png', obs_2_img[...,::-1])

#goal = env.sample_goal()
#obs_img = 255*goal['desired_goal'].reshape(3, 480, 480).transpose()
#cv2.imwrite('a.png', obs_img)
#cv2.imshow('window', obs_img)
#cv2.waitKey()
