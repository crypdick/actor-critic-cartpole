import os

import gym
import numpy as np
import tensorflow as tf
import tflearn

# use correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use correct GPU

ENV_NAME = 'CartPole-v0'
VIDEO_DIR = './results/videos/'
TENSORBOARD_RESULTS_DIR_PREFIX = './results/tensorboard/b/'

STATE_DIM = 4
ACTION_DIM = 1
ACTION_BOUND = 1  # 0 to 1
ACTION_SPACE = [1, 0]  # it is important actions are in this order due to how we set up out log-probability func

N_EPISODES = 1000

# discount factor
# the relevant window into the future is about 10 timesteps. (1 *0.7)^10 shrinks to 3% after 10 timesteps.
DISCOUNT_FACTOR = 0.99  # aka gamma

# https://www.youtube.com/watch?v=oPGVsoBonLM
# policy gradient goal: maximize E[Reward|policy*]

''' gradient estimator:
for generic E[f(x)] where x is sampled ~ prob dist p(x|theta), we want to compute the gradient wrt parameter theta:
grad_wrt_x(E_x(f(x)))

we don't need to know anything about f(x), just sample from the distribution.

'''


def make_hparam_string(learning_rate, n_neurons, batch_size):
    # dropout = "dropout={}".format(use_dropout)
    # fc_param = "fc=2" if use_two_fc else "fc=1"
    neurons = "n_neurons={}".format(n_neurons)
    batch = "batch_size={}".format(batch_size)
    return "lr_%.0E,n_%s,%s," % (learning_rate, neurons, batch)


class Policy(object):
    def __init__(self, sess, *args):
        self.sess = sess

        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_bounds = ACTION_BOUND
        self.action_space = ACTION_SPACE

    def choose_action(self, probabilities):
        prob_per_action = [probabilities, 1 - probabilities]  # compensate for having one probability
        choice = int(np.random.choice(ACTION_SPACE, 1, p=prob_per_action))
        return choice

    def get_trainable_params(self):
        return [None]

    def calc_gradient(self, *args):
        return np.zeros(1), None

    def run_optimization_step(self, _):
        pass


class RandomPolicy(Policy):
    """A policy to benchmark the policy gradient against. takes a random action at every timestep."""

    def __init__(self, *args, sess=None):
        super().__init__(sess, *args)
        self.__name__ = 'random'

    def calc_action_probabilities(self, _):
        choice = np.random.choice(2)  # spits out 0 or 1
        if choice:  # go right
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])
        return probabilities


class ContrarianPolicy(Policy):
    """A policy to benchmark the policy gradient against. whatever direction the pole is tilting, it will try to push it
     back to upright"""

    def __init__(self, *args, sess=None):
        super().__init__(sess)

    def calc_action_probabilities(self, state):
        state = state[0]  # list in a list
        theta = state[1]
        if theta >= 0.:  # action space is flipped for the policy gradient; first index is right, second index is left
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])

        return probabilities


class PolicyGradient(Policy):
    """policy gradient. actor and critic are children."""

    def __init__(self, *args, sess=None):
        super().__init__(sess)

        self.actor = ActorNetwork(self.sess, *args)

    def calc_action_probabilities(self, current_state):
        action_probabilities, summaries = self.sess.run([self.actor.action_probability_op, self.actor.summary_op1],
                                                        feed_dict={self.actor.input_states: current_state})
        return action_probabilities, summaries

    def get_trainable_params(self):
        tparams = self.sess.run(self.actor.trainable_net_params)
        return tparams

    def calc_gradient(self, ep_states, ep_fake_labels, ep_discounted_rewards):
        gradients, summary = self.sess.run([self.actor.gradient_wrt_params, self.actor.summary_op2],
                                           feed_dict={self.actor.input_states: ep_states,
                                                      self.actor.ep_fake_action_labels: ep_fake_labels,
                                                      self.actor.reward_signal: ep_discounted_rewards})
        return gradients, summary

    def run_optimization_step(self, gradient_buffer):
        self.sess.run(self.actor.optimize_step, feed_dict={self.actor.W1_gradients: gradient_buffer[0],
                                                           self.actor.W2_gradients: gradient_buffer[1]})


class ActorNetwork(object):
    """picks our actions at a given state"""

    def __init__(self, sess, learning_rate, n_neurons):
        self.sess = sess
        self.learning_rate = learning_rate
        self.n_units = n_neurons
        # self.use_two_fc = use_two_fc
        # self.use_dropout = use_dropout
        self.input_states, self.action_probability_op, self.summary_op1 = self.mk_action_predictor_net()
        self.trainable_net_params = tf.trainable_variables()
        self.n_trainable_params = len(self.trainable_net_params)

        (self.reward_signal, self.ep_fake_action_labels, self.loss, self.gradient_wrt_params, self.optimizer,
         self.W1_gradients, self.W2_gradients, self.optimize_step, self.summary_op2) = self.mk_optimizer()

    def mk_action_predictor_net(self):
        """neural network that outputs probabilities of each action"""
        with tf.name_scope('action_predictor_net'):
            input_states = tflearn.input_data(shape=[None, STATE_DIM], name='input_state')
            actor_net = tflearn.fully_connected(input_states, self.n_units, activation='relu',
                                                weights_init='xavier',
                                                name='fc1')
            tflearn.summaries.add_trainable_vars_summary([actor_net.W], name_prefix='fc1')
            actor_net = tflearn.fully_connected(actor_net, ACTION_DIM, weights_init='xavier',
                                                # bias=True, bias_init='truncated_normal',
                                                name='score')
            tflearn.summaries.add_trainable_vars_summary([actor_net.W], name_prefix='fc4')
            # output probabilities
            actor_net = tflearn.activation(actor_net, activation='sigmoid', name='probability')
            tflearn.summaries.add_activations_summary([actor_net], name_prefix='sigmoid')
            summary_op = tf.summary.merge_all()

            return input_states, actor_net, summary_op

    def mk_optimizer(self):
        with tf.name_scope('optimizer'):
            reward_signal = tf.placeholder(tf.float32, [None, 1], name='reward_signal')
            # self.action_taken = tf.placeholder("float", [None, 1], name='action_taken')
            ep_fake_action_labels = tf.placeholder(tf.float32, [None, 1], name="fake_actions")
            # self.action_probabilities = tf.placeholder("float", [None, ACTION_DIM], name='action_predictions')

            # If action_taken is 0, then the first term is eliminated, and it becomes tf.log(probability) .
            # If action_taken is 1 instead then it becomes tf.log(1-probability)
            #  we ensure that the term inside the log is never negative, and that we can compute the gradient
            # no matter which action is taken
            prob_of_other_action = ((ep_fake_action_labels * (1 - self.action_probability_op)) +
                                    ((1 - ep_fake_action_labels) * self.action_probability_op))
            log_likelihood_wrong_action = tf.log(prob_of_other_action)
            loss = -tf.reduce_mean(log_likelihood_wrong_action * reward_signal)

            gradient_wrt_params = tf.gradients(loss, self.trainable_net_params)
            # for grad in gradient_wrt_params:
            #     print(grad)
            #     tflearn.summaries.add_gradients_summary([grad], name_prefix='ep_gradients')
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            W1_gradients = tf.placeholder(tf.float32, name="batch_grad1")
            W2_gradients = tf.placeholder(tf.float32, name="batch_grad2")
            batch_gradients = [W1_gradients, W2_gradients]

            optimize_step = optimizer.apply_gradients(zip(batch_gradients, self.trainable_net_params))

            summary_op = tf.summary.merge_all()

            return reward_signal, ep_fake_action_labels, loss, gradient_wrt_params, optimizer, W1_gradients, \
                W2_gradients, optimize_step, summary_op


def run_episodes(policy, sess, batch_size, hparam):
    """runs all the episodes"""

    sess.run(tf.global_variables_initializer())

    env = gym.make(ENV_NAME)
    # env = gym.wrappers.Monitor(env, VIDEO_DIR+hparam, force=True)
    render_env = False

    writer = tf.summary.FileWriter(TENSORBOARD_RESULTS_DIR + hparam, sess.graph)
    summary_ops, summary_vars = build_summaries()

    episode_i = 1
    batch_reward_sum = 0.

    # make a gradient array w same shape as params, fill with zeros
    gradient_buffer = policy.get_trainable_params()
    for ix, grad in enumerate(gradient_buffer):
        gradient_buffer[ix] = np.zeros_like(grad)

    while episode_i <= N_EPISODES:
        done = False

        states, actions, action_probabilities, fake_action_labels, rewards = [], [], [], [], []
        summaries = []

        env_state = env.reset()

        while not done:  # ep not finished
            # only render if we're close to solving
            if batch_reward_sum / batch_size >= 199 or render_env is True:
                env.render()
                render_env = True

            current_state = np.reshape(env_state, [1, STATE_DIM])

            action_prob, summary = policy.calc_action_probabilities(current_state)
            summaries.append(summary)
            action_probabilities.append(action_prob)
            # print(action_probabilities)
            # action = policy.choose_action(action_prob)
            action = 1 if np.random.uniform() < action_prob else 0
            # actions.append(action)

            fake_label = 1 if action == 0 else 0  # a "fake label"
            fake_action_labels.append(fake_label)

            # for some reason putting this early breaks training
            # this adds state from previous loop, so we dont have to worry about handling final timestep
            states.append(current_state)

            env_state, reward, done, info = env.step(action)
            rewards.append(reward)
            batch_reward_sum += reward

            #  custom reward function to manually promote low thetas and x around 0
            # x, theta = future_state
            # low_theta_bonus = -100. * (theta ** 2.) + 1.  # reward of 1 at 0 rads, reward of 0 at +- 0.1 rad/6 deg)
            # # center_pos_bonus = -1 * abs(0.5 * x) + 1  # bonus of 1.0 at x=0, goes down to 0 as x approaches edge
            # reward += low_theta_bonus

        # this block executes after episode is done
        episode_i += 1
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        ep_states = np.vstack(states)
        ep_fake_labels = np.vstack(fake_action_labels)
        ep_rewards = np.vstack(rewards)

        ep_discounted_rewards = calc_discounted_rewards(ep_rewards)

        gradients, summary = policy.calc_gradient(ep_states, ep_fake_labels, ep_discounted_rewards)
        summaries.append(summary)

        thetas = ep_states[:, 2]
        thetas = np.reshape(thetas, np.shape(ep_rewards))

        summary_str = sess.run(summary_ops, feed_dict={
            summary_vars[0]: len(ep_rewards),
            summary_vars[1]: thetas,
            summary_vars[2]: ep_discounted_rewards
        })
        summaries.append(summary_str)
        for s in summaries:
            try:
                writer.add_summary(s, episode_i)
            except TypeError:  # Random and Contrarian policies don't actually make tf summaries
                pass
        writer.flush()

        for ix, grad in enumerate(gradients):
            gradient_buffer[ix] += grad  # gradients add onto themselves, variances smooth out

        # If we have completed enough episodes, then update the policy network with our gradients.
        if episode_i % batch_size == 0:
            policy.run_optimization_step(gradient_buffer)
            # after updating, reset gradient buffer for next batch
            for ix, grad in enumerate(gradient_buffer):
                gradient_buffer[ix] = np.zeros_like(grad)

            # Give a summary of how well our network is doing for each batch of episodes.
            print("E {:d} Average reward for episode in last batch: {:.1f}".format(episode_i,
                                                                                   batch_reward_sum / batch_size))

            if batch_reward_sum / batch_size > 200:
                print("Task solved in", episode_i, 'episodes!')
                break

            batch_reward_sum = 0  # reset for next batch


def calc_discounted_rewards(rewards):
    """rewards are reward at that specific timestep plus discounted value of future rewards in that trajectory"""
    discounted_rewards = np.zeros_like(rewards)
    running_rewards = 0.
    for t in reversed(range(0, rewards.size)):  # step backwards in time from the end of the episode
        running_rewards = rewards[t] + DISCOUNT_FACTOR * running_rewards
        discounted_rewards[t] += running_rewards
    # feature scaling/normalizing using standardization. reward vec will always have 0 mean and variance 1
    # otherwise, early rewards become astronomical as the running reward gets added to each previous ts
    # scaled rewards are much smaller, which makes it more stable for the neural network to approximate.
    # since we subtract the mean, we also see that the lat
    discounted_rewards -= discounted_rewards.mean()
    discounted_rewards /= discounted_rewards.std()
    return discounted_rewards


def build_summaries():
    """for doing a post-mortem in Tensorboard"""
    episode_time = tf.Variable(0.)
    a = tf.summary.scalar("Episode length", episode_time)
    thetas = tf.placeholder("float", [None, 1])
    b = tf.summary.histogram("Distribution of thetas", thetas)
    # episode_avg_reward = tf.Variable(0.)
    # c = tf.summary.scalar("Average Reward per action", episode_avg_reward)
    # losses = tf.placeholder("float", [None, 1])
    # d = tf.summary.histogram('Losses', losses)
    ep_rewards = tf.placeholder("float", [None, 1])
    e = tf.summary.histogram("rewards", ep_rewards)

    summary_vars = [episode_time, thetas, ep_rewards]
    summary_ops = tf.summary.merge([a, b, e])

    return summary_ops, summary_vars


def main(_):
    """parameter sweep to find the best model"""
    for i in range(1):  # repeat everything a few times to get statistical to get a sense of how stable models are
        global TENSORBOARD_RESULTS_DIR
        TENSORBOARD_RESULTS_DIR = TENSORBOARD_RESULTS_DIR_PREFIX + "{}/".format(i)  # put each iteration in a different folder for tensorboard
        for learning_rate in [1e-1]:#np.linspace(1e-1, 1e-6, 20):
            for use_two_fc in [False]:
                for n_neurons in [120]:#np.linspace(10,200, 20, dtype=np.int):
                    for use_dropout in [False]:  # dropout always made things worse
                        for batch_size in [1]:
                            # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                            hparam = make_hparam_string(learning_rate, n_neurons, batch_size)
                            print('Starting run for %s' % hparam)
                            tf.reset_default_graph()
                            with tf.Session() as sess:
                                policies = {'random': RandomPolicy, 'contrarian': ContrarianPolicy,
                                            'policy_gradient': PolicyGradient}
                                policy = policies['policy_gradient'](learning_rate, n_neurons, sess=sess)
                                run_episodes(policy, sess, batch_size, hparam)



if __name__ == '__main__':
    tf.app.run()
