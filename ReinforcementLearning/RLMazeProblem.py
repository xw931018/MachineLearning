# ACTIONS = ['Left', 'Right', 'Up', 'Down']
ACTIONS = ['←', '→', '↑', '↓']
threshold = 0.05


def moveOneStep(current, action, shape):
    current = np.copy(current)
    if action == '↑':  # 'Up':
        current[0] = max(current[0] - 1, 0)
    if action == '↓':  # 'Down':
        current[0] = min(current[0] + 1, shape[1] - 1)
    if action == '←':  # 'Left':
        current[1] = max(current[1] - 1, 0)
    if action == '→':  # 'Right':
        current[1] = min(current[1] + 1, shape[1] - 1)
    return current


# def findPrevState(current, action, shape):
#     current = np.copy(current)
#     if action == 'Up':
#         current[0] = min(current[0] + 1, shape[1] - 1)
#     if action == 'Down':
#         current[0] = max(current[0] - 1, 0)
#     if action == 'Left':
#         current[1] = min(current[1] + 1, shape[1] - 1)
#     if action == 'Right':
#         current[1] = max(current[1] - 1, 0)
#     return current

class MazeAgent:
    def __init__(self, rewards, start=None, end=None, gamma=0.95):
        '''start, end: tuples indicating the start and end of the maze
           rewards: the reward matrix. NaN value means unpassable wall in the maze.
        '''
        self._maze = rewards
        self._shape = np.array(rewards.shape)
        self._start = start if start is not None else np.array((0, 0))
        self._end = end if end is not None else self._shape - np.array((1, 1))
        self.value = np.zeros(rewards.shape)
        self.actionValue = np.zeros(list(rewards.shape) + [4])
        self.gamma = gamma

    def _initialize(self, newStart=None):
        '''Initialize the agent to the starting point of the maze'''
        self._current = np.copy(self._start) if newStart is None else newStart
        # Randomize policies
        self._policy = np.array(random.choices(ACTIONS, k=self._shape[0] * self._shape[1])).reshape(self._shape)
        self._initialPolicy = np.copy(self._policy)

    def _walkThroughEpsilon(self, maxIter=50000, policy=None):
        '''This is a brute force try where we assume we know the starting point and end point of the maze.
           Note that there is no information about the rewards at all in the Monte-Carlo methods
        '''
        count = 0
        policy = np.copy(self._policy) if policy is None else policy

        while not np.array_equal(self._current, self._end) and count < maxIter:
            self._prevCurrent = np.copy(self._current)
            epsilon = np.random.uniform()
            if epsilon < threshold:
                policy[self._current[0], self._current[1]] = random.choice(ACTIONS)
            self._current = moveOneStep(self._current, policy[self._current[0], self._current[1]],
                                        self._shape)
            if self._maze[self._current[0], self._current[1]] == -np.infty:
                self._current = np.copy(self._prevCurrent)
            count = count + 1

        # logger.info('Number of iterations: {}'.format(count))
        return policy

    def _policyEvaluation(self, iterations=100, policy=None):
        '''Given a policy (self._policy), we'd like to compute the value function V_pi'''
        policy = self._policy if policy is None else policy
        self.value = np.zeros(rewards.shape)
        for i in range(iterations):
            valueCurrent = np.copy(self.value)
            for row in range(self._shape[0]):
                for col in range(self._shape[1]):
                    if self._maze[row, col] == -np.infty:
                        continue
                    action = policy[row, col]
                    nextState = moveOneStep(np.array([row, col]), action, self._shape)
                    if self._maze[nextState[0], nextState[1]] == -np.infty:
                        nextState = np.array([row, col])
                    self.value[row, col] = self._maze[nextState[0], nextState[1]] + self.gamma * valueCurrent[
                        nextState[0], nextState[1]]
        return self.value

    def actionValueMonteCarlo(self, numberOfPaths=1000, alpha=0.05):
        '''Use MC to approximate action value function q_pi(s, a) given a policy pi'''
        self.actionValue = np.zeros(list(rewards.shape) + [4])

        for i in range(numberOfPaths):
            self._current = np.copy(self._start)
            policyInitial = np.copy(self._initialPolicy)
            actionValueCurrent = np.copy(self.actionValue)
            newPolicy = self._walkThroughEpsilon(policy=policyInitial)
            newValue = self._policyEvaluation(iterations=500, policy=newPolicy)
            for row in range(self._shape[0]):
                for col in range(self._shape[1]):
                    action = newPolicy[row, col]
                    actionIndex = ACTIONS.index(action)
                    self.actionValue[row, col, actionIndex] = actionValueCurrent[row, col, actionIndex] + alpha * (
                                newValue[row, col] - actionValueCurrent[row, col, actionIndex])
        return self.actionValue

    def policyImprovement(self, policy):
        policy = np.copy(policy)
        for row in range(self._shape[0]):
            for col in range(self._shape[1]):
                policy[row, col] = ACTIONS[np.argmax(self.actionValue[row, col])]
        self._policy = np.copy(policy)
        self._initialPolicy = np.copy(policy)
        return policy

    def policyIteration(self, policy, mcPaths=100):

        actionValue = self.actionValueMonteCarlo(numberOfPaths=mcPaths)
        policy = self.policyImprovement(policy)
        return policy, actionValue

    def _visualizeOneMatrix(self, matrix, actionMatrix, title='', formatter='{:0.01f}', colormap='hot'):
        fig, ax = plt.subplots(figsize=(12, 8))

        plt.imshow(matrix, cmap=colormap, )
        plt.title(title)
        plt.colorbar()

        displayMatrix = np.array([formatter.format(val) + '\n' + action
                                  for val, action in zip(matrix.ravel(), actionMatrix.ravel())]).reshape(matrix.shape)
        for (i, j), z in np.ndenumerate(displayMatrix):
            ax.text(j, i, z, ha='center', va='center', size=15)

    def visualize(self):
        '''Visualize the maze, value function and policy'''
        ## Visualize maze
        self._visualizeOneMatrix(self._maze, actionMatrix=self._initialPolicy, title='Maze and initial policy')
        ## Visualize policy and value
        self._visualizeOneMatrix(self.value, actionMatrix=self.policy, title='Value function and policy after learning')

    def run(self, iterations=500, mcPaths=10):
        '''Run policy iteration algorithm by Monte-Carlo action-value evaluation'''
        for i in range(iterations):
            policy, actionValue = self.policyIteration(self._initialPolicy, mcPaths=mcPaths)
        return policy, actionValue

def main():
    maze0 = np.array([[-1, -1, -1, -1, -1],
                      [-1, -np.infty, -np.infty, -np.infty, -1],
                      [-1, -1, -1, -1, -1],
                      [-10, -np.infty, -1, -np.infty, -np.infty],
                      [-1, -1, -1, -1, 10]])

    maze1 = np.array([[1., -np.infty, 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., -np.infty, 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., -np.infty, 1., 1., 1., 1.],
                     [-np.infty, -np.infty, 1., -np.infty, -np.infty, 1., -np.infty, 1., 1., 1.],
                     [1., 1., -np.infty, 1., -np.infty, 1., -np.infty, -np.infty, -np.infty, 1.],
                     [1., 1., -np.infty, 1., -np.infty, 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., -np.infty, -np.infty, -np.infty, -np.infty],
                     [1., -np.infty, -np.infty, -np.infty, -np.infty, -np.infty, 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., -np.infty, 1., 10]
                     ])

    agent0 = MazeAgent(maze0)

    agent0._initialize()
    pol0, actValue0 = agent0.run(iterations = 500, mcPaths = 1)
    agent0.visualize()

    agent1 = MazeAgent(maze1)

    agent1._initialize()
    pol1, actValue1 = agent1.run(iterations = 500, mcPaths = 1)
    agent1.visualize()