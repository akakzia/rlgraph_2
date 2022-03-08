import numpy as np
import os
import itertools

from env import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def objects_distance(x, y):
    """
    A function that returns the euclidean distance between two objects x and y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x - y)

def is_above(x, y):
    """
    A function that returns whether the object x is above y
    """
    assert x.shape == y.shape
    if np.linalg.norm(x[:2] - y[:2]) < 0.05 and 0.06 > x[2] - y[2] > 0.03:
        return 1.
    else:
        return -1.

class FetchManipulateEnvContinuous(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, num_blocks, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range, predicate_threshold,
            distance_threshold, initial_qpos, reward_type, predicates, goals_on_stack_probability=1.0, 
            allow_blocks_on_stack=True, all_goals_always_on_stack=False
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            num_blocks: number of block objects in environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type: Incremental Rewards are used by default
            predicates: 'above' and 'close'
            p_coplanar: The probability of initializing all blocks without stacks
            p_stack_two: The probability of having exactly one stack of two given that there is at least one stack
            p_grasp: The probability of having a block grasped at initialization
        """

        self.target_goal = None
        self.binary_goal = None
        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.predicate_threshold = predicate_threshold
        self.predicates = predicates
        self.num_predicates = len(self.predicates)
        self.reward_type = reward_type

        self.goals_on_stack_probability = goals_on_stack_probability
        self.allow_blocks_on_stack = allow_blocks_on_stack
        self.all_goals_always_on_stack = all_goals_always_on_stack

        # self.goal_size = 0

        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        self.location_record = None
        self.location_record_write_dir = None
        self.location_record_prefix = None
        self.location_record_file_number = 0
        self.location_record_steps_recorded = 0
        self.location_record_max_steps = 2000

        super(FetchManipulateEnvContinuous, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        self.min_max_x = (self.initial_gripper_xpos[0] - self.obj_range, self.initial_gripper_xpos[0] + self.obj_range)
        self.min_max_y = (self.initial_gripper_xpos[1] - self.obj_range, self.initial_gripper_xpos[1] + self.obj_range)

    # Heatmap Generation
    # ----------------------------

    def set_location_record_name(self, write_dir, prefix):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        self.flush_location_record()
        self.location_record_write_dir = write_dir
        self.location_record_prefix = prefix
        self.location_record_file_number = 0

        return True

    def flush_location_record(self, create_new_empty_record=True):
        if self.location_record is not None and self.location_record_steps_recorded > 0:
            write_file = os.path.join(self.location_record_write_dir,"{}_{}".format(self.location_record_prefix,
                                                                           self.location_record_file_number))
            np.save(write_file, self.location_record[:self.location_record_steps_recorded])
            self.location_record_file_number += 1
            self.location_record_steps_recorded = 0

        if create_new_empty_record:
            self.location_record = np.empty(shape=(self.location_record_max_steps, 3), dtype=np.float32)

    def log_location(self, location):
        if self.location_record is not None:
            self.location_record[self.location_record_steps_recorded] = location
            self.location_record_steps_recorded += 1

            if self.location_record_steps_recorded >= self.location_record_max_steps:
                self.flush_location_record()

    # GoalEnv methods
    # ----------------------------

    def sub_goal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3:(i + 1) * 3].shape == goal_a[..., (i+1) * 3:(i + 2) * 3].shape

        return [
            np.linalg.norm(goal_a[..., i*3:(i+1)*3] - goal_b[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = achieved_goal[..., -3:]
        sub_goals = goal[..., :-3]

        distances = [
            np.linalg.norm(gripper_pos - sub_goals[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]

        return np.all([d > self.distance_threshold*2 for d in distances], axis=0)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # print(self.reward_type)
        distances = self.sub_goal_distances(achieved_goal, goal)
        if self.reward_type == 'incremental':
            # Using incremental reward for each block in correct position
            reward = np.sum([(d < self.distance_threshold).astype(np.float32) for d in distances], axis=0)
            reward = np.asarray(reward)
            # np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_type == 'sparse':
            reward = np.min([-(d > self.distance_threshold).astype(np.float32) for d in distances], axis=0)
            reward = np.asarray(reward)
            # np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_type == 'dense':
            d = goal_distance(achieved_goal, goal)
            if min([-(d > self.distance_threshold).astype(np.float32) for d in distances]) == 0:
                return 0
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.log_location(location=self.sim.data.get_site_xpos('robot0:grip'))
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _is_close(self, d):
        """ Return the value of the close predicate for a given distance between two pairs """
        if d < self.predicate_threshold:
            return 1.
        else:
            return -1

    def _get_configuration(self, positions):
        """
        This functions takes as input the positions of the objects in the scene and outputs the corresponding semantic configuration
        based on the environment predicates
        """
        close_config = np.array([])
        above_config = np.array([])
        if "close" in self.predicates:
            object_combinations = itertools.combinations(positions, 2)
            object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

            close_config = np.array([self._is_close(distance) for distance in object_rel_distances])
        if "above" in self.predicates:
            object_permutations = itertools.permutations(positions, 2)

            above_config = np.array([is_above(obj[0], obj[1]) for obj in object_permutations])
        
        res = np.concatenate([close_config, above_config])
        return res

    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        achieved_goal = []

        for i in range(self.num_blocks):
            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])

            achieved_goal = np.concatenate([
                achieved_goal, object_i_pos.copy()
            ])

        achieved_goal_binary = self._get_configuration(achieved_goal.reshape(self.num_blocks, 3))

        achieved_goal = np.squeeze(achieved_goal)

        if self.binary_goal is None:
            self.binary_goal = np.zeros(9)

        if self.target_goal is None:
            self.target_goal = self._sample_goal()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.target_goal.copy(),
            'achieved_goal_binary': achieved_goal_binary.copy(),
            'desired_goal_binary': self.binary_goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # print("sites offset: {}".format(sites_offset[0]))
        for i in range(self.num_blocks):

            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = self.target_goal[i*3:(i+1)*3] - sites_offset[i]

        self.sim.forward()

    def reset(self, goal=None, biased_init=True):
        self.binary_goal = goal

        self.target_goal, goals, number_of_goals_along_stack = self._sample_goal(goal, return_extra_info=True)
        # self.target_goal = self.sample_continuous_goal_from_binary_goal(goal)

        self.sim.set_state(self.initial_state)
        if biased_init:
            if np.random.uniform() < 0.8:
                goal_class = 0
            else:
                goal_class = np.random.randint(1, 5)
            initial_object_positions = self._sample_goal(goal_class, return_extra_info=False)
            initial_object_positions = initial_object_positions.reshape((self.num_blocks, 3))
            np.random.shuffle(initial_object_positions)
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[:3] = initial_object_positions[i]

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

        # If evaluation mode, generate blocks on the table with no stacks
        else:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                        self.obj_range,
                                                                                        size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

        self.sim.forward()
        obs = self._get_obs()

        return obs

    def reset_goal(self, goal=None, biased_init=True):
        return self.reset(goal=goal, biased_init=biased_init)


    def _sample_goal(self, goal=None, return_extra_info=False):
        if goal is not None:
            number_of_goals_along_stack = goal + 1
        else: 
            number_of_goals_along_stack = 1

        goal0 = None
        first_goal_is_valid = False
        while not first_goal_is_valid:
            goal0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if self.num_blocks > 4:
                if np.linalg.norm(goal0[:2] - self.initial_gripper_xpos[:2]) < 0.09:
                    continue
            first_goal_is_valid = True

        goal0 += self.target_offset
        goal0[2] = self.height_offset

        goals = [goal0]

        prev_x_positions = [goal0[:2]]
        # goal_in_air_used = False
        for i in range(self.num_blocks - 1):
            if i < number_of_goals_along_stack - 1:
                goal_i = goal0.copy()
                goal_i[2] = self.height_offset + (0.05 * (i + 1))
            else:
                goal_i_set = False
                goal_i = None
                while not goal_i_set or np.any([np.linalg.norm(goal_i[:2] - other_xpos) < 0.06 for other_xpos in prev_x_positions]):
                    goal_i = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                    goal_i_set = True

                goal_i += self.target_offset
                goal_i[2] = self.height_offset

                # if np.random.uniform() < 0.2 and not goal_in_air_used:
                #     goal_i[2] += self.np_random.uniform(0.03, 0.1)
                #     goal_in_air_used = True

            prev_x_positions.append(goal_i[:2])
            goals.append(goal_i)
        # Do not consider gripper pos
        # goals.append([0.0, 0.0, 0.0])
        if not return_extra_info:
            return np.concatenate(goals, axis=0).copy()
        else:
            return np.concatenate(goals, axis=0).copy(), goals, number_of_goals_along_stack

    def _is_success(self, achieved_goal, desired_goal):

        distances = self.sub_goal_distances(achieved_goal, desired_goal)
        if sum([-(d > self.distance_threshold).astype(np.float32) for d in distances]) == 0:
            return True
        else:
            return False


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()

            # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _grasp(self, i):
        obj = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
        obj[:3] = self.sim.data.get_site_xpos('robot0:grip')
        self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), obj.copy())
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.0240)
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.0240)