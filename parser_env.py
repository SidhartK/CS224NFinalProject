import numpy as np
import gym
from gym import spaces

from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter
from parser_transitions import PartialParse

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'

class ParserEnv(gym.Env):
# class ParserEnv():

	'''2D Navigation Environment Object Centric Graph Representation'''
	metadata = {'render.modes': ['human']}

	def __init__(self, dataset_type='train', reduced=True, seed=None):
		# obj_set=[((0, 2), 0.5), ((1.2, 3.4), 0.2), ((5, 0, 1), ((2.3, 5.6), 0.6), ((4.1, 4.143), 0.139), ((5, 3.3), 0.26)], goal_set=[((8.2, 6.719), 1, 5), ((2, 8.3), 1, 12), ((7.2, 3.4), 0.5, 10)]):
		super(ParserEnv, self).__init__()
		assert(dataset_type in ['train', 'dev', 'test'])

		self._seed = seed
		self.rng = np.random.default_rng(seed)

		self.parser, self.embeddings, train_set, dev_set, test_set = load_and_preprocess_data(reduced)
		self.dataset = train_set if dataset_type == 'train' else (dev_set if dataset_type == 'dev' else test_set)

		self.ex_idx = 0
		self.dataset_len = len(self.dataset)
		self.ex = self.dataset[self.ex_idx]
		n_words = len(self.ex['word']) - 1
		self.sentence = [j + 1 for j in range(n_words)]
		self.pp = PartialParse(self.sentence)

		self.time_step = 0
		self.prev_UAS = 0.0

		self.action_space = spaces.Discrete(3)
		self.reward_range = (-1, 1)
		self.observation_dims = self._get_cur_state().shape
		self.observation_space = spaces.Box(low=np.min(self.embeddings), high=np.max(self.embeddings), shape=self.observation_dims, dtype=np.float32)

	def _get_cur_state(self):
		w = self.parser.extract_features(self.pp.stack, self.pp.buffer, self.pp.dependencies, self.ex)
		features = np.array(w).astype('int32')
		X = self.embeddings[features].flatten()
		return X


	def reset(self):
		self.ex_idx = (self.ex_idx + 1) % self.dataset_len
		self.ex = self.dataset[self.ex_idx]
		n_words = len(self.ex['word']) - 1
		self.sentence = [j + 1 for j in range(n_words)]
		self.pp = PartialParse(self.sentence)
		self.time_step = 0
		self.prev_UAS = 0.0

		state = self._get_cur_state()
		return state


	def step(self, action):
		legal_labels = self.parser.legal_labels(self.pp.stack, self.pp.buffer)
		if legal_labels[action] == 0:
			if legal_labels[2] != 0:
				action = 2
			elif legal_labels[1] != 0:
				action = 1
			else:
				action = 0

		transition = "S" if action == 2 else ("LA" if action == 0 else "RA")
		self.pp.parse_step(transition)
		head = [-1] * (len(self.ex['word']))
		UAS = all_tokens = 0.0
		t_idx = []
		for h, t, in self.pp.dependencies:
			head[t] = h
			t_idx.append(t)

		for pred_h, gold_h, pos in \
				zip(head[1:], self.ex['head'][1:], self.ex['pos'][1:]):
				assert self.parser.id2tok[pos].startswith(P_PREFIX)
				pos_str = self.parser.id2tok[pos][len(P_PREFIX):]
				if (self.parser.with_punct) or (not punct(self.parser.language, pos_str)):
					UAS += 1 if pred_h == gold_h else 0
					all_tokens += 1
		UAS = UAS/all_tokens if all_tokens > 0 else 0.0
		reward = UAS - self.prev_UAS
			

		state = self._get_cur_state()
		done = (len(self.pp.stack) == 1 and len(self.pp.buffer) == 0)
		return state, reward, done, {}



	def render(self, mode='human', close=False):
		print("Sentence: ", self.sentence)
		print("Partial Parse: ", self.pp.stack, self.pp.buffer, self.pp.dependencies)
		print("Previous UAS: ", self.prev_UAS)



if __name__ == '__main__':
	env = ParserEnv()
	time_steps = []
	rewards = []
	for i in range(10):
		print("---------------------------------------------\n\n")
		done = False
		j = 0
		env.reset()
		ep_rew = 0
		while not done:
			action = np.random.randint(3)
			obs, reward, done, _ = env.step(action)
			j += 1
			ep_rew += reward
			print(f"Episode: {i}, Time Step: {j}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
			# if j > 100000:
				# break
		time_steps.append(j)
		rewards.append(reward)
	print(f"Average time: {sum(time_steps)/len(time_steps)}")
	print(f"Average total reward: {sum(rewards)/len(rewards)}")









