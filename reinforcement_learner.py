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

STATE_DIM = 4
ACTION_DIM = 1
ACTION_PROB_DIMS = 2
ACTION_BOUND = 1  # 0 to 1
ACTION_SPACE = [0, 1]

N_EPISODES = 10
MAX_EP_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001  # 0.0001
CRITIC_LEARNING_RATE = 0.001
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

    def calc_action_probabilities(self, _):
        choice = np.random.choice(2)  # spits out 0 or 1
        if choice:  # go right
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])

        return probabilities


class ContrarianPolicy(Policy):
    def __init__(self, sess):
        super().__init__(sess)

    def calc_action_probabilities(self, state):
        state = state[0]  # list in a list
        theta = state[1]
        if theta >= 0.:
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])

        return probabilities


class PolicyGradient(Policy):
    """AKA policy gradient"""

    def __init__(self, sess):
        super().__init__(sess)
        init = tf.initialize_all_variables()
        self.sess.run(init)

        self.actor = ActorNetwork()
        self.critic = CriticNetwork()



        # fxn: use critic action gradient to figure out how to update online actor
        # self.training_gradients = tf.gradients(self.policy_network, self.all_net_params,
        #                                    -self.action_gradient_from_critic)

        # fxn: apply gradient to online actor
        # self.optimize_online_actor = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE). \
        #    apply_gradients(zip(self.actor_gradients, self.online_weights), name='online_policy_optimizer')

    def calc_action_probabilities(self, observed_states):
        action_probabilities = self.sess.run(self.actor.action_predictor,
                                             feed_dict={self.actor.input_states: observed_states})
        action_probabilities = action_probabilities[0]  # reduce depth by 1
        # print("tf action probs", action_probabilities)
        return action_probabilities

    def predict_rewards(self, observed_states, observed_actions):
        predicted_rewards = self.sess.run(self.critic.reward_predictor, feed_dict={
            self.in_states: observed_states,
            self.in_actions: observed_actions
        })
        return predicted_rewards

    def update_policy(self, observed_states, observed_actions, observed_rewards):
        """when episode concludes, lets update our actors and critics"""
        predicted_rewards = self.predict_rewards(observed_states, observed_actions)
        self.critic.optimize_critic_network(observed_states, observed_actions, observed_rewards)
        gradient_wrt_actions = self.critic.calc_action_gradient(observed_states, observed_actions)
        self.actor.optimize_actor_net(observed_states, gradient_wrt_actions)



class ActorNetwork(object):
    def __init__(self, sess):
        self.sess = sess
        self.n_units = 20
        self.input_states, self.action_predictor = self.mk_action_predictor_net()
        self.all_net_params = tf.trainable_variables()
        self.num_trainable_vars = len(self.all_net_params)

        self.gradient_wrt_actions, self.actor_optimizer = self.mk_actor_optimizer()

    def mk_action_predictor_net(self):
        """neural network that outputs probabilities of each action"""
        input_states = tflearn.input_data(shape=[None, STATE_DIM], name='input_state')
        actor_net = tflearn.fully_connected(input_states, self.n_units, activation='relu',
                                            weights_init='truncated_normal',
                                            name='hidden1')
        actor_net = tflearn.fully_connected(actor_net, ACTION_PROB_DIMS, activation='softmax',
                                            weights_init='truncated_normal', name='output_action_probabilities')

        return input_states, actor_net

    def mk_actor_optimizer(self):
        """optimizer for actor network"""
        # action gradient will be given to this by the critic network
        action_gradient_from_critic = tf.placeholder(tf.float32, [None, ACTION_DIM])
        # apply action gradient to network
        actor_gradients = tf.gradients(self.action_predictor, self.all_net_params, -action_gradient_from_critic)
        optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE). \
            apply_gradients(zip(actor_gradients, self.all_net_params))
        return action_gradient_from_critic, optimizer

    def optimize_actor_net(self, in_states, action_gradient):
        """runs optimizer using gradient from critic network"""
        self.sess.run(self.actor_optimizer,
                      feed_dict={self.input_states: in_states, self.gradient_wrt_actions: action_gradient})


class CriticNetwork(object):
    def __init__(self):
        self.n_units = 50

        self.input_states, self.input_actions, self.reward_predictor = self.mk_reward_predictor_network()
        self.observed_rewards, self.critic_optimizer = self.mk_reward_network_optimizer()
        self.action_gradient_calculator = self.mk_action_gradient_func()


    def mk_reward_predictor_network(self):
        """
        represents our belief of what rewards should be
        neural network that outputs 
        the None in the shapes allows the critic to output a reward tensor that is whatever length any given
        episode might be
        """
        input_states = tflearn.input_data(shape=[None, STATE_DIM], name='input_states')
        input_actions = tflearn.input_data(shape=[None, ACTION_DIM], name='input_actions')
        r_net = tflearn.fully_connected(input_states, self.n_units, activation='relu', name='hidden1')

        # Add the action tensor in the 2nd hidden layer
        # these two lines are hacks just to get weights and biases
        t1 = tflearn.fully_connected(r_net, self.n_units)
        t2 = tflearn.fully_connected(input_actions, self.n_units)

        r_net = tflearn.activation(tf.matmul(r_net, t1.W) +
                                                  tf.matmul(input_actions, t2.W) + t2.b, activation='relu',
                                                  name='combine_state_actions')

        # linear layer connected to 1 output representing Q(s,a)
        # TODO: does this have to be only one unit?
        r_net = tflearn.fully_connected(r_net, 1, weights_init='truncated_normal', name='output_rewards')

        return input_states, input_actions, r_net

    def mk_reward_network_optimizer(self):
        observed_rewards = tf.placeholder(tf.float32, [None, 1])
        predicted_rewards = tf.placeholder(tf.float32, [None, 1])


        # Define loss and optimization Op
        loss = tflearn.mean_square(observed_rewards, predicted_rewards)
        optimizer = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(loss)
        return observed_rewards, optimizer


    def optimize_critic_network(self, observed_states, observed_action, observed_rewards):  # note: replaced predicted_q_value with sum of mixed rewards
        """update our predictions based on observations"""
        self.sess.run([self.reward_predictor, self.critic_optimizer], feed_dict={
            self.input_states: observed_states,
            self.input_actions: observed_action,
            self.reward_predictor: observed_rewards
        })

    def mk_action_gradient_func(self):
        """critisize our actor's predictions
        given actions, predict the gradient of the rewards wrt the actions"""
        # Get the gradient of the net w.r.t. the action
        action_grad_calculator = tf.gradients(self.reward_predictor, self.input_actions)
        return action_grad_calculator

    def calc_action_gradient(self, states, actions):
        """given states and the actions the actor took, give actor a gradient for the actor to improve its
        actions wrt the critic's reward belief network"""
        action_gradient = self.sess.run(self.action_gradient_calculator, feed_dict={
            self.input_states: states,
            self.input_actions: actions
        })

        return action_gradient


def train(sess, env, policy):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(TENSORBOARD_RESULTS_DIR, sess.graph)
    total_times = []
    total_rewards = []

    for episode in range(N_EPISODES):
        # print("starting ep", episode)

        states, actions, rewards = [], [], []

        current_state = env.reset()
        states.append(current_state)
        last_timestep_i = 0

        for ts in range(MAX_EP_STEPS):
            if RENDER_ENV:
                env.render()
            action_probabilities = policy.calc_action_probabilities(np.reshape(current_state, (1, STATE_DIM)))
            # print(action_probabilities)
            action = policy.choose_action(action_probabilities)
            actions.append(action)
            future_state, reward, done, info = env.step(action)
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

        # update policy
        policy.update_policy(states, actions, discounted_rewards)


        print("episode {} | total reward {} | avg reward {} | time alive {}".format(episode, total_reward,
                                                                                    total_reward / total_time,
                                                                                    total_time))

    return total_times, total_rewards


def calc_discounted_rewards(rewards, total_time):
    discounted_rewards = np.zeros_like(rewards)
    running_rewards = 0
    for i in range(total_time - 1, -1, -1):  # step backwards in time from the end of the episode
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
