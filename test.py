import multiworld
import gym
import mujoco_py
import cv2
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import *
import time
import random

imsize = 480
multiworld.register_all_envs()
#env = gym.make('SawyerPickupMultiobj-v0')
#env = gym.make('SawyerPickupWideEnv-v0')
env = gym.make('SawyerMultiObj-v0')
#env = gym.make('SawyerPushNIPS-v0'

env = ImageEnv(
    env,
    imsize = imsize,
    init_camera=sawyer_pusher_camera_upright_v3,
    transpose=True,
    normalize=True,

)
i = 0

env.reset()
for j in range(0, 2000000):
    action = np.array([random.random()-.5, random.random()-.5, random.random()-.5])
    obs = env.step(action)[0]['image_observation']
    obs_img = 255*obs.reshape(3, 480, 480).transpose()
    cv2.imwrite('/home/lab/imgs/obs' + str(j) + '.png', obs_img[...,::-1])


#goal = env.sample_goal()
#obs_img = 255*goal['desired_goal'].reshape(3, 480, 480).transpose()
#cv2.imwrite('a.png', obs_img)
#cv2.imshow('window', obs_img)
#cv2.waitKey()
