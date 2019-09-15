from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.core.serializable import Serializable

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import *
from multiworld.envs.mujoco.util.create_xml import create_object_xml, create_root_xml, clean_xml
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

import random
import copy
import multiworld

BASE_DIR = '/'.join(str.split(multiworld.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/multiworld/envs/assets/sawyer_xyz/'

class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    INIT_HAND_POS = np.array([0, 0.6, 0.2])

    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.05, 0.65, 0.2),

            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,
            frame_skip = 50,
            obj_init_positions=((0, 0.6, 0.02),),
            random_init=False,

            num_objects=1,
            fix_goal=False,

            mocap_low=(-0.13, 0.57, 0.04),
            mocap_high=(0.08, 0.73, 0.2),
            action_scale=0.02,

            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
            goal_low=None,
            goal_high=None,
            reset_free=False,
            filename='sawyer_pick_and_place_multiobj.xml',
            object_mass=1,
            # object_meshes=['Bowl', 'GlassBowl', 'LotusBowl01', 'ElephantBowl', 'RuggedBowl'],
            object_meshes=None,
            obj_classname = None,
            block_height=0.04,
            block_width = 0.04,
            cylinder_radius = 0.015,
            finger_sensors=False,
            maxlen=0.06,
            minlen=0.01,
            preload_obj_dict=None,
            hide_goal_markers=False,
            oracle_reset_prob=0.0,
            presampled_goals=None,
            num_goals_presampled=2,
            p_obj_in_hand=1,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)


        base_filename = asset_base_path + filename
        friction_params = (1, 1, 2)
        self.obj_stat_prop = create_object_xml(base_filename, num_objects, object_mass,
                                              friction_params, object_meshes, finger_sensors,
                                               maxlen, minlen, preload_obj_dict, obj_classname,
                                               block_height, block_width, cylinder_radius)

        gen_xml = create_root_xml(base_filename)
        MujocoEnv.__init__(self, gen_xml, frame_skip=frame_skip)
        self.reset_mocap_welds()

        clean_xml(gen_xml)

        self.mocap_low = mocap_low
        self.mocap_high = mocap_high
        self.action_scale = action_scale


        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)

        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.reward_type = reward_type
        self.random_init = random_init
        self.p_obj_in_hand = p_obj_in_hand
        self.indicator_threshold = indicator_threshold

        self.obj_init_z = obj_init_positions[0][2]
        self.obj_init_positions = np.array(obj_init_positions)
        self.last_obj_pos = self.obj_init_positions[0]

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None
        self.reset_free = reset_free
        self.oracle_reset_prob = oracle_reset_prob

        self.obj_ind_to_manip = 0

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
            dtype=np.float32
        )
        self.hand_space = Box(
            self.hand_low,
            self.hand_high,
            dtype=np.float32
        )
        self.gripper_and_hand_and_obj_space = Box(
            np.hstack(([0.0], self.hand_low, obj_low)),
            np.hstack(([0.04], self.hand_high, obj_high)),
            dtype=np.float32
        )
        self.num_objects = num_objects
        self.maxlen = maxlen

        self.observation_space = Dict([
            ('observation', self.gripper_and_hand_and_obj_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),
            ('state_observation', self.gripper_and_hand_and_obj_space),
            ('state_desired_goal', self.hand_and_obj_space),
            ('state_achieved_goal', self.hand_and_obj_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])
        self.hand_reset_pos = np.array([0, .6, .2])

        if presampled_goals is not None:
            self._presampled_goals = presampled_goals
            self.num_goals_presampled = len(list(self._presampled_goals.values)[0])
        else:
            # presampled_goals will be created when sample_goal is first called
            self._presampled_goals = None
            self.num_goals_presampled = num_goals_presampled
        self.num_goals_presampled = 10
        self.picked_up_object = False
        self.train_pickups = 0
        self.eval_pickups = 0
        self.cur_mode = 'train'
        self.reset()

    def mode(self, name):
        if 'train' not in name:
            self.oracle_reset_prob = 0.0
            self.cur_mode = 'train'
        else:
            self.cur_mode = 'eval'

    def viewer_setup(self):
        sawyer_pick_and_place_camera_slanted_angle(self.viewer.cam)

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_gripper_pos(self):
        return np.array([self.data.qpos[7]])

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_xy_action(self, xy_action, fixed_z):
        delta_z = fixed_z - self.data.mocap_pos[0, 2]
        xyz_action = np.hstack((xy_action, delta_z))
        self.set_xyz_action(xyz_action)

    def step(self, action):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        for i in range(self.num_objects):
            x = 7 + i * 7 + 1
            y = 10 + i * 7 + 1
            qpos[x:y] = np.clip(qpos[x:y], self.obj_low, self.obj_high)
        self.set_state(qpos, qvel)

        self.set_xyz_action(action[:3])
        self.do_simulation(action[3:])
        new_obj_pos = self.get_obj_pos()
        # if the object is out of bounds and not in the air, move it back
        if new_obj_pos[2] < .05:
            new_obj_pos[0:2] = np.clip(
                new_obj_pos[0:2],
                self.obj_low[0:2],
                self.obj_high[0:2]
            )
        elif new_obj_pos[2] > .05:
            if not self.picked_up_object:
                if self.cur_mode == 'train':
                    self.train_pickups += 1
                else:
                    self.eval_pickups += 1
                self.picked_up_object = True
        self.set_object_xy(self.obj_ind_to_manip, new_obj_pos)
        #self._set_obj_xyz(new_obj_pos)
        self.last_obj_pos = new_obj_pos.copy()
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def set_reset_pos(self, pos):
        self.hand_reset_pos = pos

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        gripper = self.get_gripper_pos()
        flat_obs = np.concatenate((e, b))
        flat_obs_with_gripper = np.concatenate((gripper, e, b))
        hand_goal = self._state_goal[:3]
        return dict(
            observation=flat_obs_with_gripper,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs_with_gripper,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=e,
            proprio_achieved_goal=e,
            proprio_desired_goal=hand_goal,
        )

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        obj_goal = self._state_goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        obj_distance = np.linalg.norm(obj_goal - self.get_obj_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_obj_pos()
        )
        return dict(
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            hand_and_obj_distance=hand_distance+obj_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            obj_success=float(obj_distance < self.indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold
            ),
            total_pickups=self.train_pickups if self.cur_mode == 'train' else self.eval_pickups,
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_obj_pos(self):
        mujoco_id = self.model.body_names.index('object' + str(self.obj_ind_to_manip))
        return self.data.body_xpos[mujoco_id].copy()
        #return self.data.get_body_xpos('object' + str(self.obj_ind_to_manip)).copy()

    def get_object_pos(self, id):
        mujoco_id = self.model.body_names.index('object' + str(id))
        return self.data.body_xpos[mujoco_id].copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (
            goal[3:]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (
                -1000
            )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        #qpos[8:11] = pos.copy()
        #qvel[8:15] = 0
        qpos[7 + 7 * self.obj_ind_to_manip + 1: 7 + 7 * self.obj_ind_to_manip + 3 + 1] = pos.copy()
        qvel[7 + 7 * self.obj_ind_to_manip + 1: 7 + 7 * self.obj_ind_to_manip + 1 + 7] = 0
        self.set_state(qpos, qvel)

    def set_object_xy(self, i, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        x = 7 + i * 7 + 1
        y = 10 + i * 7 + 1
        z = 14 + i * 7 + 1
        if len(pos) < 3:
            qpos[x:y] = np.hstack((pos.copy(), np.array([0.04])))
        else:
            qpos[x:y] = pos
        qpos[y:z] = np.array([1, 0, 0, 0])
        x = 7 + i * 6 + 1
        y = 13 + i * 6 + 1
        qvel[x:y] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        while True:
            pos = np.zeros((self.num_objects , 2)) #position contains hand/obj positions
            #pos[0] = self.hand_reset_pos[:2] #first elem of pos is hand
            #pos[self.obj_ind_to_manip] = goal[0][:2] #keep randomly initialized obj_to_manip

            for i in range(self.num_objects): #re-arrange all other obj positions to make sure no touching
                r = np.random.uniform(self.obj_low[:2], self.obj_high[:2])
                pos[i] = r
            touching = []
            for i in range(self.num_objects):
                for j in range(i):
                    t = np.linalg.norm(pos[i] - pos[j]) <= self.maxlen
                    touching.append(t)
            if not any(touching):
                break
        for i in range(self.num_objects):
            self.set_object_xy(i, pos[i])

        self.do_simulation(None) #Adding in simulation frame skips to avoid collision
    #    self._set_obj_xyz(pos[1:])
        self.set_goal(self.sample_goal())
        #self._set_goal_marker(self._state_goal)
        self.picked_up_object = False

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        for i in range(self.num_objects):
            x = 7 + i * 7 + 1
            y = 10 + i * 7 + 1
            qpos[x:y] = np.clip(qpos[x:y], self.obj_low, self.obj_high)
        self.set_state(qpos, qvel)
        self.do_simulation(np.array([-1]), n_frames=50) #Adding in simulation frame skips to avoid collision


        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_reset_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def set_to_goal(self, goal, is_hand_att=True):
        """
        This function can fail due to mocap imprecision or impossible object
        positions.
        """
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(np.array([-1]))
        #error = self.data.get_site_xpos('endeffector') - hand_goal
        #print(error)
        #corrected_obj_pos = state_goal[3:] + error
        #corrected_obj_pos[2] = max(corrected_obj_pos[2], self.obj_init_z)
        self._set_obj_xyz(state_goal[3:])
        if np.linalg.norm(self.data.get_site_xpos('endeffector') - 
        if corrected_obj_pos[2] > .05:
            action = np.array(1)
        else:
            action = np.array(1 - 2 * np.random.choice(2))

        for _ in range(10):
            self.do_simulation(action)
        self.sim.forward()
#        else:
#            state_goal = goal['state_desired_goal']
#            hand_goal = state_goal[:3]
#            for _ in range(30):
#                self.data.set_mocap_pos('mocap', hand_goal)
#                self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
#                self.do_simulation(np.array([-1]))
#            error = self.data.get_site_xpos('endeffector') - hand_goal
#            corrected_obj_pos = state_goal[3:] + error
#            self._set_obj_xyz(state_goal[3:])
#            action = np.array(-1)
#            for _ in range(10):
#                self.do_simulation(action)
#            self.sim.forward()


    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def sample_goals(self, batch_size):
        self._presampled_goals = \
                corrected_state_goals(
                    self,
                    self.generate_uncorrected_env_goals(
                        self.num_goals_presampled
                    )
                )
        idx = np.random.randint(0, self.num_goals_presampled, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals


    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        obj_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        obj_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        obj_distances = np.linalg.norm(obj_goals - obj_pos, axis=1)
        hand_and_obj_distances = hand_distances + obj_distances
        touch_distances = np.linalg.norm(hand_pos - obj_pos, axis=1)
        touch_and_obj_distances = touch_distances + obj_distances

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'obj_distance':
            r = -obj_distances
        elif self.reward_type == 'obj_success':
            r = -(obj_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_obj_distance':
            r = -hand_and_obj_distances
        elif self.reward_type == 'touch_and_obj_distance':
            r = -touch_and_obj_distances
        elif self.reward_type == 'hand_and_obj_success':
            r = -(
                hand_and_obj_distances < self.indicator_threshold
            ).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'touch_distance',
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'touch_success',
            'hand_distance',
            'obj_distance',
            'hand_and_obj_distance',
            'total_pickups',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_env_state_inh(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state_inh(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()

    def get_env_state(self):
        base_state = self.get_env_state_inh()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        self.set_env_state_inh(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)

    def generate_uncorrected_env_goals(self, num_goals):
        """
        Due to small errors in mocap, moving to a specified hand position may be
        slightly off. This is an issue when the object must be placed into a given
        hand goal since high precision is needed. The solution used is to try and
        set to the goal manually and then take whatever goal the hand and object
        end up in as the "corrected" goal. The downside to this is that it's not
        possible to call set_to_goal with the corrected goal as input as mocap
        errors make it impossible to rereate the exact same hand position.

        The return of this function should be passed into
        corrected_image_env_goals or corrected_state_env_goals
        """
        if self.fix_goal:
            goals = np.repeat(self.fixed_goal.copy()[None], num_goals, 0)
        else:
            goals = np.random.uniform(
                self.hand_and_obj_space.low,
                self.hand_and_obj_space.high,
                size=(num_goals, self.hand_and_obj_space.low.size),
            )
            num_objs_in_hand = int(num_goals * self.p_obj_in_hand)
            if num_goals == 1:
                num_objs_in_hand = int(np.random.random() < self.p_obj_in_hand)

            bs = [[0, 0]]
            for i in range(self.num_objects): #put all obj positions inside pos
                b = self.get_object_pos(i)[:2]
                bs.append(b)
            for j in range(0, num_objs_in_hand):
                while True:
                    bs[0] = np.random.uniform(self.hand_low[:2], self.hand_high[:2])
                    touching = []
                    for i in range(1, self.num_objects + 1):
                        if i-1 != self.obj_ind_to_manip:
                            t = np.linalg.norm(bs[i] - bs[0]) <= self.maxlen
                            touching.append(t)
                    if not any(touching):
                        goals[j][0:2] = bs[0] #set goals[j] to new randomly init. position
                        break

            # Put object in hand
            goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
            goals[:num_objs_in_hand, 4] -= 0.038
        #    goals[:num_objs_in_hand, 5] += 0.01
            #print(goals)
            goals[:num_objs_in_hand, 2] -= 0.005

            pos = np.zeros((self.num_objects + 1, 2)) #placeholders for collision detection


            bs = []
            for i in range(self.num_objects): #put all obj positions inside pos
                b = self.get_object_pos(i)[:2]
                bs.append(b)

            pos[1:] = bs
            r = self.obj_ind_to_manip #only care about changing one object, obj_to_manip
            for j in range(num_objs_in_hand, len(goals)):
                pos[0] = goals[j][:2] #first elem. of pos is hand posiiton
                while True:
                    bs[r] = np.random.uniform(self.hand_low[:2], self.hand_high[:2]) #new puck pos.
                    touching = []
                    for i in range(self.num_objects + 1):
                        if i != r: #only compare hand+other objs
                            t = np.linalg.norm(pos[i] - bs[r]) <= self.maxlen
                            touching.append(t)
                    if not any(touching):
                        goals[j][3:5] = bs[r] #set goals[j] to new randomly init. position
                        break
            # Put object one the table (not floating)
            goals[num_objs_in_hand:, 5] = self.obj_init_z #all z positions set on table for num_obs*prob_hand:onwards
            return {
                'desired_goal': goals,
                'state_desired_goal': goals,
                'proprio_desired_goal': goals[:, :3]
            }

def corrected_state_goals(pickup_env, pickup_env_goals):
    pickup_env._state_goal = np.zeros(6)
    goals = pickup_env_goals.copy()
    num_goals = len(list(goals.values())[0])
    original_config = pickup_env._get_obs()['state_observation']
    num_obj_in_hand = int(pickup_env.num_goals_presampled * pickup_env.p_obj_in_hand)

    for idx in range(num_goals):
        obj_in_hand = False
        if idx < num_obj_in_hand:
            obj_in_hand = True
        pickup_env.set_to_goal(
            {'state_desired_goal': goals['state_desired_goal'][idx]}, obj_in_hand
        )
        corrected_state_goal = pickup_env._get_obs()['achieved_goal']
        corrected_proprio_goal = pickup_env._get_obs()['proprio_achieved_goal']

        goals['desired_goal'][idx] = corrected_state_goal
        goals['proprio_desired_goal'][idx] = corrected_proprio_goal
        goals['state_desired_goal'][idx] = corrected_state_goal
    pickup_env._reset_hand()
    pickup_env.set_object_xy(pickup_env.obj_ind_to_manip, original_config[4:])

    return goals

def corrected_image_env_goals(image_env, pickup_env_goals):
    """
    This isn't as easy as setting to the corrected since mocap will fail to
    move to the exact position, and the object will fail to stay in the hand.
    """

    image_env.wrapped_env._state_goal = np.zeros(6)
    goals = pickup_env_goals.copy()
    num_obj_in_hand = int(image_env.num_goals_presampled * image_env.p_obj_in_hand)

    num_goals = len(list(goals.values())[0])
    goals = dict(
        image_desired_goal=np.zeros((num_goals, image_env.image_length)),
        desired_goal=np.zeros((num_goals, image_env.image_length)),
        state_desired_goal=np.zeros((num_goals, 6)),
        proprio_desired_goal=np.zeros((num_goals, 3))
    )
    original_config = image_env._get_obs()['state_observation']
    for idx in range(num_goals):
        obj_in_hand = False
        if idx < num_obj_in_hand:
            obj_in_hand = True
        if idx % 100 == 0:
            print(idx)
        image_env.set_to_goal(
            {'state_desired_goal': pickup_env_goals['state_desired_goal'][idx]}, obj_in_hand
        )
        corrected_state_goal = image_env._get_obs()['state_achieved_goal']
        corrected_proprio_goal = image_env._get_obs()['proprio_achieved_goal']
        corrected_image_goal = image_env._get_obs()['image_achieved_goal']

    #    image_env.set_to_goal()

        goals['image_desired_goal'][idx] = corrected_image_goal
        goals['desired_goal'][idx] = corrected_image_goal
        goals['state_desired_goal'][idx] = corrected_state_goal
        goals['proprio_desired_goal'][idx] = corrected_proprio_goal
    image_env._reset_hand()

    image_env.set_object_xy(image_env.obj_ind_to_manip, original_config[4:])
    return goals

def get_image_presampled_goals(image_env, num_presampled_goals):
    image_env.reset()
    pickup_env = image_env.wrapped_env
    image_env_goals = corrected_image_env_goals(
        image_env,
        pickup_env.generate_uncorrected_env_goals(num_presampled_goals)
    )
    return image_env_goals
