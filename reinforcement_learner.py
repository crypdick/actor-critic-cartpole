import numpy as np
import tensorflow as tf
import gym
from operator import itemgetter
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=oPGVsoBonLM
# policy gradient goal: maximize E[Reward|policy*]

# start: randomly generate weights

''' gradient estimator:
for generic E[f(x)] where x is sampled ~ prob dist p(x|theta), we want to compute the gradient wrt parameter theta:
grad_wrt_x(E_x(f(x)))

we don't need to know anything about f(x), just sample from the distribution.

'''



def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    function stolen from TF MNIST example: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py#L66-L76"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def policy_gradient():
    with tf.variable_scope("policy"):
        weights = tf.get_variable("weights", [4, 2])
        state = tf.placeholder("float", [None, 4], name='state')
        actions = tf.placeholder("float", [None, 2], name='actions')
        advantages = tf.placeholder("float", [None, 1], name='advantages')
        Wx = tf.matmul(state, weights, name='Wx')
        probabilities = tf.nn.softmax(Wx, name='probabilities')
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer


def reward_gradient():
    with tf.variable_scope("reward"):
        with tf.name_scope('x'):
            state = tf.placeholder("float", [None, 4], name='state')

        new_rewards = tf.placeholder("float", [None, 1], name='new_rewards')
        with tf.name_scope('hidden_1'):
            with tf.name_scope('weights'):
                weights1 = tf.get_variable("w1", [4, 10])
                variable_summaries(weights1)
            with tf.name_scope('biases'):
                biases1 = tf.get_variable("b1", [10])
                variable_summaries(biases1)
            with tf.name_scope('Wx_plus_b'):
                hidden_1 = tf.nn.relu(tf.matmul(state, weights1) + biases1, name='hidden_1')
                tf.summary.histogram('pre_activations', hidden_1)
        with tf.name_scope('output_layer'):
            with tf.name_scope('weights'):
                w2 = tf.get_variable("w2", [10, 1])
            with tf.name_scope('biases'):
                b2 = tf.get_variable("b2", [1])
            with tf.name_scope('Wx_plus_b'):
                output_activations = tf.matmul(hidden_1, w2) + b2  # TODO: pass through activation func?  rename to preactivated
        residuals = output_activations - new_rewards  # TODO: new rewards? or predicted? wat
        loss = tf.nn.l2_loss(residuals, name='loss')
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return output_activations, state, new_rewards, optimizer, loss
"""
# dropout example from https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
"""


'''
TODO: make several classes which inherit from a masterclass. each class is a separate policy: the random agent,
the agent which always pushed towards the center, and the policy gradient
'''
class Episode():
    def __init__(self, env, policy_grad, value_grad, sess, trials_per_episode=400, render=False):
        self.env = env
        if render is True:
            self.env.render()  # show animation window
        # import pdb; pdb.set_trace()
        self.pl_calculated, self.pl_state, self.pl_actions, self.pl_advantages, self.pl_optimizer = policy_grad
        self.vl_calculated, self.vl_state, self.vl_newvals, self.vl_optimizer, self.vl_loss = value_grad
        self.total_episode_reward = 0
        self.states = []
        self.actions = []
        self.advantages = []
        self.transitions = []
        self.updated_rewards = []
        self.action_space = list(range(2))
        self.current_state = None
        self.metadata = None
        # TODO: if plotting, return metadata: histograms of each episode

    def run_episode(self):
        full_state = self.env.reset()
        self.current_state = list(itemgetter(0,2)(full_state))
        timesteps_per_trial = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        print("ts per trial", timesteps_per_trial)
        thetas = []
        for ts in range(timesteps_per_trial):

            # calculate policy
            obs_vector = np.expand_dims(self.current_state, axis=0)  # shape (4,) --> (1,4)
            probs = sess.run(self.pl_calculated, feed_dict={self.pl_state: obs_vector})  # probability of both actions
            # draw action 0 with probability P(0), action 1 with P(1)
            action = self._select_action(probs[0])

            # take the action in the environment
            old_observation = self.current_state
            full_state, reward, done, info = env.step(action)
            self.current_state = list(itemgetter(0,2)(full_state))
            x, theta = self.current_state

            # custom rewards to encourage staying near center and having a low rate of theta change
            low_theta_bonus = -30.*(theta**2.) + 2. # reward of 2 at 0 rads, reward of 0 at +- 0.2582 rad/14.8 deg)
            center_pos_bonus = -1*abs(0.5*x)+1  # bonus of 1.0 at x=0, goes down to 0 as x approaches edge
            reward += center_pos_bonus * low_theta_bonus

            # store whole situation
            self.states.append(self.current_state)
            action_taken = np.zeros(2)
            action_taken[action] = 1
            self.actions.append(action_taken)
            self.transitions.append((old_observation, action, reward))
            self.total_episode_reward += reward
            thetas.append(np.abs(self.current_state[2]))

            if done:
                #print("Episode finished after {} timesteps".format(t + 1))
                break

        # now that we're done with episode, assign credits with discounted rewards
        print(np.max(thetas))
        for ts, transition in enumerate(self.transitions):
            obs, action, reward = transition

            # calculate discounted monte-carlo return
            future_reward = 0
            n_future_timesteps = len(self.transitions) - ts
            decrease = 1
            decreasing_decay_rate = 0.97
            for future_ts in range(1, n_future_timesteps):
                future_reward += self.transitions[ts + future_ts][2] * decrease
                decrease = decrease * decreasing_decay_rate
            obs_vector = np.expand_dims(obs, axis=0)
            old_future_reward = sess.run(self.vl_calculated, feed_dict={self.vl_state: obs_vector})[0][0]

            # advantage: how much better was this action than normal
            self.advantages.append(future_reward - old_future_reward)

            # update the value function towards new return
            self.updated_rewards.append(future_reward)

        # update value function
        updated_r_vec = np.expand_dims(self.updated_rewards, axis=1)
        try:
            sess.run(self.vl_optimizer, feed_dict={self.vl_state: self.states, self.vl_newvals: updated_r_vec})
        except:
            print("value gradient dump")
            print(np.shape(self.vl_state), np.shape(self.states), np.shape(self.vl_newvals), np.shape(updated_r_vec))
            print("updated rew", len(self.updated_rewards))
            raise
        # real_self.vl_loss = sess.run(self.vl_loss, feed_dict={self.vl_state: states, self.vl_newvals: update_vals_vector})

        advantages_vector = np.expand_dims(self.advantages, axis=1)

        try:
            sess.run(self.pl_optimizer, feed_dict={self.pl_state: self.states, self.pl_advantages: advantages_vector, self.pl_actions: self.actions})
        except:
            print("exception dump")
            print(np.shape(self.pl_state), np.shape(self.states), np.shape(self.pl_advantages), np.shape(advantages_vector), np.shape(self.pl_actions), np.shape(self.actions))
            raise

        return self.total_episode_reward

    def _select_action(self, probabilities):
        '''
        :param action_space: possible actions
        :param probabilities: probs of selecting each action
        :return: selected_action

        e.g. if action space is [0,1], probabilities are [.3, .7], draw is 0.5:
        thresh levels = .3, 1
        draw <= thresh ==> [False, True]
        return action_space[1]
        '''
        random_draw = np.random.uniform(0, 1)
        cumulative_probability = 0
        probability_thresholds = []
        for p in probabilities:
            cumulative_probability += p
            probability_thresholds.append(cumulative_probability)
        under_thresh = np.where(np.array(probability_thresholds) >= random_draw)[0]  # indices where true
        try:
            desired_index = under_thresh[0]  # first index where draw < thresh
        except IndexError:  # very rare case where draw between .999999999 and 1
            desired_index = len(probability_thresholds)
        selected_action = self.action_space[desired_index]

        return selected_action


if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    # flags.DEFINE_boolean('upload',False,'upload to gym (requires evironment variable OPENAI_GYM_API_KEY)')
    flags.DEFINE_string('env', 'CartPole-v0', 'gym environment')
    # flags.DEFINE_integer('train',10000,'training time between tests. use 0 for test run only')
    # flags.DEFINE_integer('test',10000,'testing time between training')

    # flags.DEFINE_bool('random',False,'use random agent')
    # flags.DEFINE_bool('tot',False,'train on test data')
    # flags.DEFINE_integer('total',1000000,'total training time')
    # flags.DEFINE_float('monitor',.05,'probability of monitoring a test episode')
    # flags.DEFINE_bool('autonorm',False,'automatically normalize observations and actions of the environemnt')

    env = gym.make(FLAGS.env)
    env = gym.wrappers.Monitor(env, 'log/cartpole-trial1', force=True)  # save trial vids
    flags.DEFINE_integer('tmax', env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'),
                         'maximum timesteps per episode')
    #print(env.action_space)  # Discrete(2), i.e. 0 or 1
    # print(env.observation_space) #  Box(4,)
    #print(env.observation_space.high)  # [  4.80000000e+00   3.40282347e+38   4.18879020e-01   3.40282347e+38]
    #print(env.observation_space.low)  # same as above with flipped signs

    policy_grad = policy_gradient()
    #print("policy_grad", policy_grad)
    #raise
    reward_grad = reward_gradient()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        reward_timeline = []
        episode = Episode(env, policy_grad, reward_grad, sess, render=False)
        n_episodes = 9000
        for i_episode in range(n_episodes):
            episode_total_reward = episode.run_episode()
            reward_timeline.append(episode_total_reward)

        # TODO save progress to resume learning weights

        '''
        TODO: make series of graphs I talked about in the capstone proposal
        '''
        plt.plot(np.arange(len(reward_timeline)), reward_timeline)
        plt.show()

        #merged = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter('/graph', sess.graph)

        # TODO: figure out how to make tf tensorboard graphs

    #gym.upload('/tmp/cartpole-experiment-1', api_key='ZZZ')