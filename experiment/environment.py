"""Partial-observation queue simulator; ground truth never enters observations."""
import numpy as np


class QueueTriage:
    # Wait, inspect CPU, inspect dependency, diagnose healthy/CPU/downstream, escalate.
    ACTIONS = ('wait', 'cpu', 'dependency', 'healthy', 'cpu_fault', 'downstream_fault', 'escalate')

    def __init__(self, seed=0, shifted=False, abstention_penalty=.5):
        self.rng = np.random.default_rng(seed)
        self.abstention_penalty = float(abstention_penalty)
        self.shifted = shifted
        self.done = True

    def reset(self, fault=None):
        self.fault = int(self.rng.integers(3)) if fault is None else fault
        self.load = float(self.rng.uniform(.55, .85) if not self.shifted else self.rng.uniform(.85, .98))
        self.cap = .4 if self.fault else 1.0
        self.queue = float(self.rng.uniform(0, .1))
        self.t = 0
        self.cpu = self.dep = 0.0
        self.cpu_seen = self.dep_seen = False
        self.checks = 0
        self.done = False
        self.choice = None
        return self.observe()

    def observe(self):
        # Fault 1 and 2 intentionally have identical passive observation distributions.
        latency = .1 + min(self.queue, 4) * .3
        return np.array([self.load, min(self.queue, 4)/4, latency,
                         min(max(latency-.6, 0), 1), self.cpu, self.dep,
                         float(self.cpu_seen), float(self.dep_seen), self.t/6], np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError(action)
        self.t += 1
        self.queue = max(0., self.queue + self.load - self.cap + self.rng.normal(0, .015))
        reward = -.05
        if action in (1, 2):
            self.checks += 1
            reward -= .15
            if action == 1:
                self.cpu = float(np.clip(self.rng.normal(.9 if self.fault == 1 else .35, .06), 0, 1))
                self.cpu_seen = True
            else:
                self.dep = float(np.clip(self.rng.normal(.9 if self.fault == 2 else .2, .06), 0, 1))
                self.dep_seen = True
        elif 3 <= action <= 5:
            self.choice = action - 3
            reward += 2. if self.choice == self.fault else -2.
            self.done = True
        elif action == 6:
            self.choice = -1
            reward -= self.abstention_penalty
            self.done = True
        if self.t >= 6 and not self.done:
            self.done = True
            self.choice = -1
            reward -= self.abstention_penalty
        return self.observe(), reward, self.done


class DreamerEnv:
    def __init__(self, seed=0, shifted=False, abstention_penalty=.5):
        import elements
        self.sim = QueueTriage(seed, shifted, abstention_penalty)
        self.obs_space = dict(vector=elements.Space(np.float32, (9,)),
                              reward=elements.Space(np.float32),
                              is_first=elements.Space(bool), is_last=elements.Space(bool),
                              is_terminal=elements.Space(bool))
        self.act_space = dict(reset=elements.Space(bool), action=elements.Space(np.int32, (), 0, 7))

    def step(self, action):
        if bool(action['reset']) or self.sim.done:
            return dict(vector=self.sim.reset(), reward=np.float32(0), is_first=True,
                        is_last=False, is_terminal=False)
        vector, reward, done = self.sim.step(int(action['action']))
        return dict(vector=vector, reward=np.float32(reward), is_first=False,
                    is_last=done, is_terminal=done)

    def close(self):
        pass
