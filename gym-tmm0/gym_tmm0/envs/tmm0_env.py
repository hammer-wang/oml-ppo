import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TMM(gym.Env):
	'''
	TMM module for calculating the reward. 
	'''

	def __init__(self):
		self.layers = []
		self.thicknesses = []
		self.current_merit = 0

	def config(self, simulator, merit_func, target):

		self.simulator = simulator # tmm simulator
		self.merit_func = merit_func
		self.target = target

	def step(self, action):

		layer, thickness = action

		if layer == 'done':
			episode_over = True
			return (self.layers, self.thicknesses), 0, True, {}
		else:
			episode_over = False

		self.layers.append(layer)
		self.thicknesses.append(thickness)
		R, T, A = self.simulator.spectrum(self.layers, [np.inf] + self.thicknesses + [np.inf])
		merit = self.merit_func(R, T, A, self.target)
		reward = merit - self.current_merit
		self.current_merit = merit

		return (self.layers, self.thicknesses), reward, episode_over, {}

	def reset(self):
		self.layers = []
		self.thicknesses = []
		self.current_merit = 0

	def render(self):

		self.simulator.spectrum(self.layers, [np.inf] + self.thicknesses + [np.inf], plot=True, title=True)