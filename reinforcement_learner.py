import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import tflearn
from operator import itemgetter  # for shrinking state size
import os

# use correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use correct GPU


ENV_NAME = 'CartPole-v0'
RENDER_ENV = True
SAVE_METADATA = True
SAVE_VIDS = True
VID_DIR = './log/videos/'
MONITOR_DIR = './results/gym_ddpg'
TENSORBOARD_RESULTS_DIR = './results/tf_results'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'

STATE_DIM = 2
ACTION_DIM = 1
ACTION_PROB_DIMS = 2
ACTION_BOUND = 1  # 0 to 1
ACTION_SPACE = [0, 1]

N_EPISODES = 10
MAX_EP_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001  # 0.0001
# Discount factor
DISCOUNT_FACTOR = 0.99  # aka gamma


# https://www.youtube.com/watch?v=oPGVsoBonLM
# policy gradient goal: maximize E[Reward|policy*]

# start: randomly generate weights

''' gradient estimator:
for generic E[f(x)] where x is sampled ~ prob dist p(x|theta), we want to compute the gradient wrt parameter theta:
grad_wrt_x(E_x(f(x)))

we don't need to know anything about f(x), just sample from the distribution.

'''



class Policy(object):
    def __init__(self, sess):
        self.sess = sess
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_bounds = ACTION_BOUND
        self.action_space = ACTION_SPACE

    def choose_action(self, probabilities):
        choice = int(np.random.choice(ACTION_SPACE, 1, p=probabilities))
        return choice


class RandomPolicy(Policy):
    def __init__(self, sess):
        super().__init__(sess)

    def predict_action(self, _inputs):
        choice = np.random.choice(2)  # spits out 0 or 1
        if choice:  # go right
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])

        return probabilities

class ContrarianPolicy(Policy):
    def __init__(self, sess):
        super().__init__(sess)

    def predict_action(self, state):
        state = state[0]  # list in a list
        theta = state[1]
        if theta >= 0.:
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])

        return probabilities


class PolicyGradient(Policy):
    '''AKA policy gradient'''
    def __init__(self, sess):
        super().__init__(sess)
        init = tf.initialize_all_variables()
        self.sess.run(init)

        self.actor = ActorNetwork()
        self.critic = CriticNetwork()


        # fxn: use critic action gradient to figure out how to update online actor
        #self.training_gradients = tf.gradients(self.policy_network, self.all_net_params,
        #                                    -self.action_gradient_from_critic)

        # fxn: apply gradient to online actor
        #self.optimize_online_actor = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE). \
        #    apply_gradients(zip(self.actor_gradients, self.online_weights), name='online_policy_optimizer')


    def predict_action(self, state):
        action_probabilities = self.sess.run(self.policy_network, feed_dict={self.input_state: state})
        action_probabilities = action_probabilities[0]  # reduce depth by 1
        #print("tf action probs", action_probabilities)
        return action_probabilities

class ActorNetwork(object):
    def __init__(self):
        self.n_units = 20
        self.input_state, self.policy_network = self.mk_actor_network()
        self.all_net_params = tf.trainable_variables()

    def mk_actor_network(self):
        """neural network that outputs probabilities of each action"""
        in_state = tflearn.input_data(shape=[None, STATE_DIM], name='input_state')
        layer1 = tflearn.fully_connected(in_state, self.n_units, activation='relu', weights_init='truncated_normal',
                                         name='hidden1')
        output_probabilities = tflearn.fully_connected(layer1, ACTION_PROB_DIMS, activation='softmax',
                                    weights_init='truncated_normal', name='out_probabilities')

        return in_state, output_probabilities

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.onnet_in_states: inputs,
            self.action_gradient: a_gradient
        })

class CriticNetwork(object):
    def __init__(self):
        pass


def train(sess, env, actor):
# Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(TENSORBOARD_RESULTS_DIR, sess.graph)
    total_times = []
    total_rewards = []

    for episode in range(N_EPISODES):
        #print("starting ep", episode)

        states, actions, rewards = [], [], []

        current_state = env.reset()
        current_state = list(itemgetter(0, 2)(current_state))
        states.append(current_state)
        last_timestep_i = 0

        for ts in range(MAX_EP_STEPS):
            if RENDER_ENV:
                env.render()
            action_probabilities = actor.predict_action(np.reshape(current_state, (1, STATE_DIM)))
            #print(action_probabilities)
            action = actor.choose_action(action_probabilities)
            actions.append(action)
            future_state, reward, done, info = env.step(action)
            future_state = list(itemgetter(0, 2)(future_state))
            rewards.append(reward)
            # print("future state ", np.shape(future_state))

            # x, theta = future_state
            # low_theta_bonus = -100. * (theta ** 2.) + 1.  # reward of 1 at 0 rads, reward of 0 at +- 0.1 rad/6 deg)
            # # center_pos_bonus = -1 * abs(0.5 * x) + 1  # bonus of 1.0 at x=0, goes down to 0 as x approaches edge
            # reward += low_theta_bonus


            current_state = future_state

            if not done:  # prevent adding future state if done (len(states) == len rewards == len actions)
                states.append(future_state)
            else:
                last_timestep_i = ts  # max /index/; len(timesteps) == max_i + 1

                break
        total_time = last_timestep_i + 1
        total_times.append(total_time)

        discounted_rewards = calc_discounted_rewards(rewards, total_time)
        total_reward = np.sum(discounted_rewards)
        total_rewards.append(total_reward)
        print("episode {} | total reward {} | avg reward {} | time alive {}".format(episode, total_reward,
                total_reward/total_time, total_time))

    return total_times, total_rewards


def calc_discounted_rewards(rewards, total_time):
    discounted_rewards = np.zeros_like(rewards)
    running_rewards = 0
    for i in range(total_time-1, -1, -1):  # step backwards in time from the end of the episode
        discounted_rewards[i] = rewards[i] + DISCOUNT_FACTOR * running_rewards
    # feature scaling/normalizing using standardization. reward vec will always have 0 mean and variance 1
    discounted_rewards -= discounted_rewards.mean()
    discounted_rewards /= discounted_rewards.std()
    return discounted_rewards

# class Episode():
#     def __init__(self, sess, env, actor):
#         self.sess = sess
#         self.env = env
#         if RENDER is True:
#             self.env.render()  # show animation window
#
#         self.actor = actor
#         # import pdb; pdb.set_trace()
#         #self.pl_calculated, self.pl_state, self.pl_actions, self.pl_advantages, self.pl_optimizer = actor
#         #self.vl_calculated, self.vl_state, self.vl_newvals, self.vl_optimizer, self.vl_loss = critic
#         self.total_episode_reward = 0
#         self.states = []
#         self.actions = []
#         self.advantages = []
#         self.transitions = []
#         self.updated_rewards = []
#         self.action_space = list(range(2))
#         self.current_state = None
#         self.metadata = None
#         # TODO: if plotting, return metadata: histograms of each episode
#
#     def run_episode(self):
#         full_state = self.env.reset()
#         self.current_state = list(itemgetter(0,2)(full_state))
#         max_episode_steps = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
#         #print("ts per trial", timesteps_per_trial)
#         thetas = []
#         for ts in range(max_episode_steps):
#
#             # calculate policy
#             obs_vector = np.expand_dims(self.current_state, axis=0)  # shape (4,) --> (1,4)
#             action_probabilities = self.actor.predict(self.current_state)
#             #probs = sess.run(self.pl_calculated, feed_dict={self.pl_state: obs_vector})  # probability of both actions
#             # draw action 0 with probability P(0), action 1 with P(1)
#             action = self._select_action(action_probabilities[0])
#
#             # take the action in the environment
#             old_observation = self.current_state
#             full_state, reward, done, info = env.step(action)
#             self.current_state = list(itemgetter(0,2)(full_state))
#             x, theta = self.current_state
#
#             # custom rewards to encourage staying near center and having a low rate of theta change
#             low_theta_bonus = -30.*(theta**2.) + 2. # reward of 2 at 0 rads, reward of 0 at +- 0.2582 rad/14.8 deg)
#             center_pos_bonus = -1*abs(0.5*x)+1  # bonus of 1.0 at x=0, goes down to 0 as x approaches edge
#             reward += center_pos_bonus * low_theta_bonus
#
#             # store whole situation
#             self.states.append(self.current_state)
#             action_taken = np.zeros(2)
#             action_taken[action] = 1
#             self.actions.append(action_taken)
#             self.transitions.append((old_observation, action, reward))
#             self.total_episode_reward += reward
#             thetas.append(np.abs(self.current_state[2]))
#
#             if done:
#                 #print("Episode finished after {} timesteps".format(t + 1))
#                 break
#
#         # # now that we're done with episode, assign credits with discounted rewards
#         # print(np.max(thetas))
#         # for ts, transition in enumerate(self.transitions):
#         #     obs, action, reward = transition
#         #
#         #     # calculate discounted return
#         #     future_reward = 0
#         #     n_future_timesteps = len(self.transitions) - ts
#         #
#         #     for future_ts in range(1, n_future_timesteps):
#         #         future_reward += self.transitions[ts + future_ts][2] * decrease
#         #         decrease = decrease * DISCOUNT_FACTOR
#         #     obs_vector = np.expand_dims(obs, axis=0)
#            # old_future_reward = sess.run(self.vl_calculated, feed_dict={self.vl_state: obs_vector})[0][0]
#
#             # # advantage: how much better was this action than normal
#             # self.advantages.append(future_reward - old_future_reward)
#             #
#             # # update the value function towards new return
#             # self.updated_rewards.append(future_reward)
#
#         # # update value function
#         # updated_r_vec = np.expand_dims(self.updated_rewards, axis=1)
#         # try:
#         #     sess.run(self.vl_optimizer, feed_dict={self.vl_state: self.states, self.vl_newvals: updated_r_vec})
#         # except:
#         #     print("value gradient dump")
#         #     print(np.shape(self.vl_state), np.shape(self.states), np.shape(self.vl_newvals), np.shape(updated_r_vec))
#         #     print("updated rew", len(self.updated_rewards))
#         #     raise
#         # # real_self.vl_loss = sess.run(self.vl_loss, feed_dict={self.vl_state: states, self.vl_newvals: update_vals_vector})
#         #
#         # advantages_vector = np.expand_dims(self.advantages, axis=1)
#         #
#         # try:
#         #     sess.run(self.pl_optimizer, feed_dict={self.pl_state: self.states, self.pl_advantages: advantages_vector, self.pl_actions: self.actions})
#         # except:
#         #     print("exception dump")
#         #     print(np.shape(self.pl_state), np.shape(self.states), np.shape(self.pl_advantages), np.shape(advantages_vector), np.shape(self.pl_actions), np.shape(self.actions))
#         #     raise
#
#         # return self.total_episode_reward
#
#     def _select_action(self, probabilities):
#         '''
#         :param action_space: possible actions
#         :param probabilities: probs of selecting each action
#         :return: selected_action
#
#         e.g. if action space is [0,1], probabilities are [.3, .7], draw is 0.5:
#         thresh levels = .3, 1
#         draw <= thresh ==> [False, True]
#         return action_space[1]
#         '''
#         choice = np.random.choice(self.action_space, 1, p=probabilities)
#         return choice


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def plot_metadata(total_times, total_rewards):
    f, axarr = plt.subplots(2, sharex=True)
    x = np.arange(N_EPISODES)
    axarr[0].plot(x, total_times)
    axarr[0].set_title('Total Times')
    axarr[0].set_ylabel("Time")
    axarr[1].scatter(x, total_rewards)
    axarr[1].set_title('Total Rewards')
    axarr[1].set_ylabel("Rewards")
    axarr[1].set_xlabel("Episode Number")

    plt.show()

def main(_):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        env = gym.make(ENV_NAME)
        policies = {'random': RandomPolicy, 'contrarian': ContrarianPolicy, 'policy_gradient': PolicyGradient}
        actor = policies['policy_gradient'](sess)
        env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)
        total_times, total_rewards = train(sess, env, actor)
        plot_metadata(total_times, total_rewards)

        # TODO save progress to resume learning weights

        # TODO: figure out how to make tf tensorboard graphs

        # gym.upload('/tmp/cartpole-experiment-1', api_key='ZZZ')


if __name__ == '__main__':
    '''scopes
    global scope should be constants, put at top
    main loop scope should have the tf session, state+action dimensionality, bounds, actor+critic networks,
    and the episode loop.
    '''

    tf.app.run()