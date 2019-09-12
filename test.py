import multiworld
import gym
import mujoco_py
import cv2
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import *
import time

imsize = 480
multiworld.register_all_envs()
env = gym.make('SawyerPickupMultiobj-v0')
#env = gym.make('SawyerPickupEnv-v0')

env = ImageEnv(
    env,
    imsize = imsize,
    init_camera=sawyer_pick_and_place_camera_slanted_angle,
    transpose=True,
    normalize=True,

)
i = 0

obs = env.reset()

action = np.array([1,1,1,1])

obs_img = 255*obs['observation'].reshape(3, 480, 480).transpose()
cv2.imwrite('obs.png', obs_img)

obs_2_img = 255*obs['desired_goal'].reshape(3, 480, 480).transpose()
cv2.imwrite('goal.png', obs_2_img)

#goal = env.sample_goal()
#obs_img = 255*goal['desired_goal'].reshape(3, 480, 480).transpose()
#cv2.imwrite('a.png', obs_img)
#cv2.imshow('window', obs_img)
#cv2.waitKey()
