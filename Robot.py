import random

"""
问题7修改建议：
### 【解释1】
* 首先给你补充一下对于 epsilon greedy 算法的解释：
* 对于 epsilon-greedy 算法，你可以参考论坛中的 这个帖子：
    Q: 如何理解 greed-epsilon 方法／如何设置 epsilon／如何理解 exploration & exploitation 权衡？
    A: (1) 我们的小车一开始接触到的 state 很少，并且如果小车按照已经学到的 qtable 执行，那么小车很有可能出错或者绕圈圈。
        同时我们希望小车一开始能随机的走一走，接触到更多的 state。(2) 基于上述原因，我们希望小车在一开始的时候不完全按照 Q learning 
        的结果运行，即以一定的概率 epsilon，随机选择 action，而不是根据 maxQ 来选择 action。然后随着不断的学习，那么我会降低这个
        随机的概率，使用一个衰减函数来降低 epsilon。(3) 这个就解决了所谓的 exploration and exploitation 的问题，在“探索”和“执行”
        之间寻找一个权衡。

### 【解释2】
* 再给你补充一下对 alpha 的解释。 alpha 是一个权衡上一次学到结果和这一次学习结果的量，如：Q = (1-alpha)*Q_old + alpha*Q_current。
* alpha 设置过低会导致机器人只在乎之前的知识，而不能积累新的 reward。一般取 0.5 来均衡以前知识及新的 reward。

### 【解释3】
* gamma 是考虑未来奖励的因子，是一个(0,1)之间的值。一般我们取0.9，能够充分地对外来奖励进行考虑。
* 实际上如果你将它调小了，你会发现终点处的正奖励不能够“扩散”到周围，也就是说，机器人很有可能无法学习到一个到达终点的策略。你可以自己尝试一下。

### 【思考】
* 你的思考「比如alpha，或许应该越来越小，也就是说每个位置的q值越来越稳定；」很棒！
* 我们知道，学习率 α 的目的是为了在更新 Q 值的同时也保留过去学到的结果，那么对于不同的 state，实际上学习的进度是不一样的。那么
    此处对所有的 state 统一设置 α，似乎并不是最优的做法。你可以考虑对每个 state 设置不同的学习率，该 state 学习完毕后其对应的 α 衰减，
    而其他 state 对应的 α 不变。可以参考 周志华 的 《机器学习》（西瓜书）中相关的内容。

### 【修改】
* 对于epsilon-greedy算法，暂时不做改变，就采用目前的方式，即：self.epsilon = self.epsilon+0.00025 if self.epsilon+0.00025 < 0.95 else 0.95
* 对于alpha，采取审阅老师的意见，针对不同的state，设置不同的alpha，并分别进行衰减，根据更新次数，衰减方式可以进行多次尝试选择效果好的衰减参数；
* 对于gamma，值过小会导致正奖励无法扩散，会使得机器人无法到达终点，0.9尽可能大的放大未来奖励；
* 思考1：游戏的首要目的是什么，如果是达到终点，那么应该想办法降低陷阱的影响，如果是累计奖励，那么对于gamma的值的设置是否有意义，或者原地踏步是最好的结果；
* 思考2：对于alpha，表示某个状态进行更新时对过去的自身以及现在的结果各选取多少，这个alpha应该随着Q本身被不断的更新而进行衰减，也就是说alpha是状态相关的；
* 思考3：对于gamma，表示对于未来奖励值的折扣率，暂时不知道如何动态处理，暂定0.9不变；
* 思考4：对于epsilon，之前已经做了衰减处理，这个值同样是针对某种状态下进行选择的概率，那么是否也应该是state相关的呢，而不是所有state共用一个；
* 思考5：对于alpha、epsilon，比如某个时间t下，两个状态s1,s2，s1被机器人走过很多次，更新了很多次Q，s2没有被走过，没更新过Q
        那么对于s1，我们可能倾向于利用，也就是高epsilon，对于现学的东西不是很重视，也就是低的alpha
        但是对于s2，由于我们没有走过这个状态，因此更倾向于探索，也就是低epsilon，但是对于现学的东西很重视，也就是高alpha
        因此对于epsilon、alpha都应该是state相关的；

问题8修改建议：
### 【问题】
* 请进一步补充你的分析：
    * 我们希望你在这里能够更详细地说明每个参数（alpha、epsilon0、epsilon下降函数、训练次数）的作用是什么，它们的变化大概
        会怎样影响运行结果，然后有目的地对小车进行调参，比较不同参数下的训练结果，并说明你使用这个参数的原因。
    * 总结出这些参数值的变化将如何影响你小车的训练结果。
    * 对比在不同的参数组合下小车的运行结果，并将结果打印出来你（你可以复制 runner.plot_results() 代码对结果多次打印。
        请至少对比三组参数组合的结果。
* 这样你的报告会更加严谨～
"""

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):
        '''
        maze:迷宫对象
        alpha:学习率
        gamma:折扣率
        epsilon0:就是贪婪度，但是会变化
        '''

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        # self.epsilon = self.epsilon0
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            pass
        else:
            # TODO 2. Update parameters when learning
            self.epsilon = self.epsilon+0.00025 if self.epsilon+0.00025 < 0.95 else 0.95

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if not state in self.Qtable.keys():
            self.Qtable[state]={'u':0.,'d':0.,'l':0.,'r':0.}

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():
            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return random.random() > self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return random.choice(self.valid_actions) # self.valid_actions[random.randint(0,len(self.valid_actions)-1)]
            else:
                # TODO 7. Return action with highest q value
                return sorted(self.Qtable[self.state].items(), key=lambda q: q[1])[-1][0]
        elif self.testing:
            # TODO 7. choose action with highest q value
            return sorted(self.Qtable[self.state].items(), key=lambda q: q[1])[-1][0]
        else:
            # TODO 6. Return random choose aciton
            return random.choice(self.valid_actions) # self.valid_actions[random.randint(0,len(self.valid_actions)-1)]

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            from_self = (1-self.alpha)*self.Qtable[self.state][action]
            # from_update = self.alpha*(r+self.gamma*(sorted(self.Qtable[next_state].items(), key=lambda q: q[1])[-1][1]))
            from_update = self.alpha*(r+self.gamma*(max(self.Qtable[next_state].values())))
            self.Qtable[self.state][action] = from_self + from_update # 此处应该是根据公式来更新，没这么简单....

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
