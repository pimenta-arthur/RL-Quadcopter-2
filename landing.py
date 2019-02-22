import numpy as np
from physics_sim import PhysicsSim

class Landing():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., touching_surface=None, threshold_velocity=None):
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
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = np.array([self.sim.pose[0], self.sim.pose[1], 0.])
        self.distance= abs(self.target_pos-self.sim.pose[:3]).sum()
        self.touching_surface = touching_surface if touching_surface is not None else 3
        self.threshold_velocity = threshold_velocity if threshold_velocity is not None else 5

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        if (self.sim.pose[3] < self.touching_surface): #position z < touching suface limit
            if (self.sim[8] < self.threshold_velocity): #velocity z < treshold velocity
                landing_reward = 100
                reward = landing_reward * (1-(self.sim.pose[3]/self.distance)**.4)
            else:
                reward = -100
        else: 
            reward = 1-(self.sim.pose[3]/self.distance)**.4

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
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state