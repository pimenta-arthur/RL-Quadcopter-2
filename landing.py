import numpy as np
from physics_sim import PhysicsSim

class Landing():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., touching_surface=None,
        threshold_velocity=None, threshold_side_distance=None):
        """Initialize a Landing object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent which is the ground
        """
        # Simulation
        self.init_pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]) if init_pose is None else np.copy(init_pose)
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = np.array([self.init_pose[0], self.init_pose[1], 0.]) # target for a successful landing
        self.distance= abs(self.target_pos-self.init_pose[:3]).sum()
        self.touching_surface = touching_surface if touching_surface is not None else 3
        self.threshold_velocity = threshold_velocity if threshold_velocity is not None else 5
        self.threshold_side_distance = threshold_side_distance if threshold_side_distance is not None else 3

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        if (self.sim.pose[2] < self.touching_surface): # position z < touching suface limit
            reward = 100 * (1-(self.sim.pose[2]/self.distance)**.4) - max(abs(self.sim.linear_accel[2]), .1)
        else: 
            reward = (1-(self.sim.pose[2]/self.distance)**.4)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
#         positive terminals
        if (self.sim.pose[2] == self.target_pos[2]): # stop if it reachs the target
            reward += 100.0  # bonus reward
            done = True
            
#         negative terminals
        if (self.sim.pose[2] > self.init_pose[2]+5): # stop if it goes up
            reward -= 100
            done = True
        # stop if it goes too much to the side 
        if (np.linalg.norm(self.init_pose[:2]-self.sim.pose[:2]) > self.threshold_side_distance): 
            reward -= 100
            done = True
        else:
            pass
            
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state