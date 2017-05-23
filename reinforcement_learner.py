import numpy as np
import tensorflow as tf
import gym
import tflearn
import os

# use correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use correct GPU

ENV_NAME = 'CartPole-v0'
RENDER_ENV = False
SAVE_VIDS = True
VIDEO_DIR = './results/videos/'
TENSORBOARD_RESULTS_DIR = './results/tensorboard/param_sweep4/'

STATE_DIM = 4
ACTION_DIM = 1
ACTION_PROB_DIMS = 2
ACTION_BOUND = 1  # 0 to 1
ACTION_SPACE = [0, 1]

N_EPISODES = 500
MAX_EP_STEPS = 200  # from CartPole env
# ACTOR_LEARNING_RATE = 0.0001
# CRITIC_LEARNING_RATE = 0.0001
# discount factor
# the relevant window into the future is about 10 timesteps. (1 *0.7)^10 shrinks to 3% after 10 timesteps.
DISCOUNT_FACTOR = 0.95  # aka gamma

# https://www.youtube.com/watch?v=oPGVsoBonLM
# policy gradient goal: maximize E[Reward|policy*]

''' gradient estimator:
for generic E[f(x)] where x is sampled ~ prob dist p(x|theta), we want to compute the gradient wrt parameter theta:
grad_wrt_x(E_x(f(x)))

we don't need to know anything about f(x), just sample from the distribution.

'''

def make_hparam_string(learning_rate, n_neurons, use_two_fc, use_dropout):
  dropout = "dropout={}".format(use_dropout)
  fc_param = "fc=2" if use_two_fc else "fc=1"
  neurons = "n_neurons={}".format(n_neurons)
  return "lr_%.0E,n_%s,%s,%s" % (learning_rate, neurons, dropout, fc_param)



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
    def __init__(self, *args, sess=None):
        super().__init__(sess)

    def calc_action_probabilities(self, _):
        choice = np.random.choice(2)  # spits out 0 or 1
        if choice:  # go right
            probabilities = np.array([0., 1.])
        else:
            probabilities = np.array([1., 0.])

        return probabilities


class ContrarianPolicy(Policy):
    def __init__(self, *args, sess=None):
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
    def __init__(self, *args, sess=None):
        super().__init__(sess)

        self.actor = ActorNetwork(self.sess, *args)
        # self.critic = CriticNetwork(self.sess, self.actor.n_trainable_params)

    def calc_action_probabilities(self, observed_states):
        action_probabilities = self.sess.run(self.actor.action_predictor,
                                             feed_dict={self.actor.input_states: observed_states})
        # fixme activations aren't in tensorboard
        tflearn.summaries.add_activations_summary([self.actor.action_predictor], name_prefix='actor_predictor_NET')
        action_probabilities = action_probabilities[0]  # reduce depth by 1
        return action_probabilities

    # def predict_rewards(self, observed_states, observed_actions):
    #     """in case we want to peek at our reward predictions"""
    #     predicted_rewards = self.sess.run(self.critic.reward_predictor, feed_dict={
    #         self.critic.input_states: observed_states,
    #         self.critic.input_actions: observed_actions
    #     })
    #     return predicted_rewards

    # def update_policy(self, observed_states, observed_actions, action_probability_timeline, observed_rewards):
    #     """when episode concludes, lets update our actors and critics"""
        # print(np.shape(observed_states), np.shape(observed_actions), np.shape(observed_rewards))
        # self.critic.optimize_critic_network(observed_states, observed_actions, observed_rewards)
        # gradient_wrt_actions = self.critic.calc_action_gradient(observed_states, observed_actions)
        # self.actor.optimize_actor_net(observed_states, gradient_wrt_actions)



class ActorNetwork(object):
    def __init__(self, sess, learning_rate, n_neurons, use_two_fc, use_dropout):
        self.sess = sess
        self.learning_rate = learning_rate
        self.n_units = n_neurons
        self.use_two_fc = use_two_fc
        self.use_dropout = use_dropout
        self.input_states, self.action_predictor = self.mk_action_predictor_net()
        self.trainable_net_params = tf.trainable_variables()
        self.n_trainable_params = len(self.trainable_net_params)

        with tf.name_scope('optimizer'):

            self.input_rewards = tf.placeholder("float", [None, 1])
            self.action_taken = tf.placeholder("float", [None, ACTION_PROB_DIMS])
            # self.loss =  tf.placeholder("float", [None, 1])
            # tf.summary.scalar('cross_entropy', self.loss)  # fixme throws InvalidArgumentError

            log_action_probability = tf.reduce_sum(self.action_taken*tf.log(self.action_predictor))
            self.loss = -log_action_probability * self.input_rewards  # could also switch to l2 loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def mk_action_predictor_net(self):
        """neural network that outputs probabilities of each action"""
        with tf.name_scope('action_predictor_net'):
            input_states = tflearn.input_data(shape=[None, STATE_DIM], name='input_state')
            actor_net = tflearn.fully_connected(input_states, self.n_units,
                                                weights_init='truncated_normal',
                                                name='fc1')
            tf.summary.histogram('fc1_DUB', actor_net.W)
            tflearn.summaries.add_trainable_vars_summary([actor_net.W, actor_net.b], name_prefix='fc1')
            if self.use_two_fc:
                actor_net = tflearn.fully_connected(actor_net,
                                                    name='fc2')
            if self.use_dropout:
                actor_net = tflearn.dropout(actor_net, 0.5, name='actor_dropout')
            actor_net = tflearn.fully_connected(actor_net, ACTION_PROB_DIMS, activation='softmax',
                                                weights_init='truncated_normal', bias=True, bias_init='truncated_normal',
                                                name='fc_output_action_probabilities')
            tflearn.summaries.add_trainable_vars_summary([actor_net.W, actor_net.b], name_prefix='final_layer')

            return input_states, actor_net

    # def mk_actor_optimizer(self):
    #     """optimizer for actor network"""
    #     with tf.name_scope('optimizer'):
    #         # action gradient will be giv
    #         # apply action gradient to network
    #         actor_gradients = tf.gradients(self.action_predictor, self.trainable_net_params, -action_gradient_from_critic)
    #         optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE). \
    #             apply_gradients(zip(actor_gradients, self.trainable_net_params))
    #         return action_gradient_from_critic, optimizer
    #
    # def optimize_actor_net(self, in_states, action_gradient):
    #     """runs optimizer using gradient from critic network"""
    #     # print(np.shape(action_gradient))
    #     self.sess.run(self.actor_optimizer,
    #                   feed_dict={self.input_states: in_states, self.gradient_wrt_actions: action_gradient})

    # def mk_trainer(self):
    #     # actions_taken = tf.placeholder(dtype=tf.float32, shape=[None, ACTION_PROB_DIMS], name="actions_taken")
    #     # action_probabilities = tf.placeholder(dtype=tf.float32, shape=[None, ACTION_PROB_DIMS], name="action_predictions")
    #     discounted_rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="discounted_rewards")
    #     # loss = tf.nn.l2_loss(actions_taken - action_probabilities)  # this gradient encourages the actions taken
    #     loss = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='loss')
    #     optimizer = tf.train.RMSPropOptimizer(ACTOR_LEARNING_RATE, decay=0.99)  # decay rate chosen arbitrarily
    #
    #     optimizer_step = optimizer.apply_gradients(action_gradients)
    #     return loss, discounted_rewards, optimizer_step

    def update_policy(self, input_states, action_probability_timeline, observed_rewards):
        """when episode concludes, lets update our actors and critics"""
        _, error_value = self.sess.run([self.optimizer, self.loss],
                                       feed_dict={self.input_states: input_states,
                                                  self.action_taken: action_probability_timeline,
                                                  self.input_rewards: observed_rewards})


# class CriticNetwork(object):
#     def __init__(self, sess, n_actor_params):
#         self.sess = sess
#         self.n_units = 200
#
#         self.input_states, self.input_actions, self.reward_predictor = self.mk_reward_predictor_network()
#         # get full list of trainable params from tf, then slice out the ones belonging to the actor network
#         self.trainable_params = tf.trainable_variables()[n_actor_params:]
#
#         self.observed_rewards, self.critic_optimizer = self.mk_reward_network_optimizer()
#         self.action_gradient_calculator = self.mk_action_gradient_func()
#
#     def mk_reward_predictor_network(self):
#         """
#         represents our belief of what rewards should be
#         neural network that outputs
#         the None in the shapes allows the critic to output a reward tensor that is whatever length any given
#         episode might be
#         """
#         with tf.name_scope('reward_predictor'):
#             input_states = tflearn.input_data(shape=[None, STATE_DIM], name='input_states')
#             input_actions = tflearn.input_data(shape=[None, ACTION_DIM], name='input_actions')
#             r_net = tflearn.fully_connected(input_states, self.n_units, activation='relu', name='hidden1')
#             tflearn.summaries.add_trainable_vars_summary([r_net.W], name_prefix='hidden1')
#
#             # Add the action tensor in the 2nd hidden layer
#             # these two lines are hacks just to get weights and biases
#             # TODO: should I change this architecture? The graph on tensorboard looks like a mess
#             t1 = tflearn.fully_connected(r_net, self.n_units, name='temp_hidden1')
#             t2 = tflearn.fully_connected(input_actions, self.n_units, name='temp_hidden2')
#
#             r_net = tflearn.activation(tf.matmul(r_net, t1.W) +
#                                        tf.matmul(input_actions, t2.W) + t2.b, activation='relu',
#                                        name='combine_state_actions')
#             # tflearn.summaries.add_activations_summary([r_net], name_prefix='combine_state_actions')
#
#             # linear layer connected to 1 output representing Q(s,a)
#             # TODO: does this have to be only one unit?
#             r_net = tflearn.dropout(r_net, 0.5, name='critic_dropout')
#             r_net = tflearn.fully_connected(r_net, 1, weights_init='truncated_normal',
#                                             bias=True, bias_init='zeros',
#                                             name='output_rewards')
#             tflearn.summaries.add_trainable_vars_summary([r_net.W, r_net.b], name_prefix='output_rewards')
#
#             return input_states, input_actions, r_net
#
#     def mk_reward_network_optimizer(self):
#         with tf.name_scope('reward_net_optimizer'):
#             observed_rewards = tflearn.input_data(shape=[None, 1], name='input_rewards')
#             loss = tflearn.mean_square(observed_rewards, self.reward_predictor)
#             optimizer = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(loss)
#             return observed_rewards, optimizer
#
#     def optimize_critic_network(self, observed_states, observed_action,
#                                 observed_rewards):  # note: replaced predicted_q_value with sum of mixed rewards
#         """update our predictions based on observations"""
#         self.sess.run([self.reward_predictor, self.critic_optimizer], feed_dict={
#             self.input_states: observed_states,
#             self.input_actions: observed_action,
#             self.observed_rewards: observed_rewards
#         })
#
#     def mk_action_gradient_func(self):
#         """critisize our actor's predictions
#         given actions, predict the gradient of the rewards wrt the actions"""
#         with tf.name_scope('action_gradient_calculator'):
#             # Get the gradient of the net w.r.t. the action
#             action_grad_calculator = tf.gradients(self.reward_predictor, self.input_actions)
#             # print(type(action_grad_calculator),np.shape(action_grad_calculator), action_grad_calculator)
#             # tflearn.helpers.summarizer.summarize_gradients([action_grad_calculator])
#             return action_grad_calculator
#
#     def calc_action_gradient(self, states, actions):
#         """given states and the actions the actor took, give actor a gradient for the actor to improve its
#         actions wrt the critic's reward belief network"""
#         action_gradient = self.sess.run(self.action_gradient_calculator, feed_dict={
#             self.input_states: states,
#             self.input_actions: actions
#         })
#         action_gradient = action_gradient[0]  # compensate for nested list
#         return action_gradient


def train(learning_rate, n_neurons, use_two_fc, use_dropout, hparam):
    tf.reset_default_graph()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(TENSORBOARD_RESULTS_DIR + hparam, sess.graph)
        policies = {'random': RandomPolicy, 'contrarian': ContrarianPolicy, 'policy_gradient': PolicyGradient}
        policy = policies['policy_gradient'](learning_rate, n_neurons, use_two_fc, use_dropout, sess=sess)
        sess.run(tf.global_variables_initializer())

        env = gym.make(ENV_NAME)
        env = gym.wrappers.Monitor(env, VIDEO_DIR+hparam, force=True)

        summary_ops, summary_vars = build_summaries()

        for episode_i in range(N_EPISODES):
            # print("starting ep", episode)

            states, actions, action_probability_timeline, rewards = [], [], [], []

            current_state = env.reset()
            states.append(current_state)
            last_timestep_i = 0

            for ts in range(MAX_EP_STEPS):
                if RENDER_ENV:
                    env.render()
                action_probabilities = policy.calc_action_probabilities(np.reshape(current_state, (1, STATE_DIM)))
                action_probability_timeline.append(action_probabilities)
                # print(action_probabilities)
                action = policy.choose_action(action_probabilities)
                actions.append(action)
                future_state, reward, done, info = env.step(action)
                # print("reward", reward)
                rewards.append(reward)
                # print("future state ", np.shape(future_state))

                #  custom reward function to manually promote low thetas and x around 0
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
            episode_length = last_timestep_i + 1
            # total_times.append(total_time)

            discounted_rewards = calc_discounted_rewards(rewards, episode_length)
            # print(discounted_rewards)

            # make our env observations into correct tensor shapes
            tensor_lengths = len(actions)
            # actions = np.reshape(actions, (tensor_lengths, 1))
            action_probability_timeline = np.reshape(action_probability_timeline, (tensor_lengths, ACTION_PROB_DIMS))
            discounted_rewards = np.reshape(discounted_rewards, (tensor_lengths, 1))

            # update policy
            policy.actor.update_policy(states, action_probability_timeline, discounted_rewards)
            # total_rewards.append(discounted_rewards.sum())

            # let's look at how our reward belief network is doing
            # reward_mse = np.mean((discounted_rewards - policy.predict_rewards(states, actions))**2)
            # reward_mses.append(reward_mse)
            #
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: episode_length,
                summary_vars[1]: discounted_rewards.sum(),
                summary_vars[2]: discounted_rewards.mean()

            })

            writer.add_summary(summary_str, episode_i)
            writer.flush()

            # print("episode {} | total reward {} | avg reward {} | time alive {}".format(
            #     episode_i,
            #     discounted_rewards.sum(),
            #     discounted_rewards.mean(),
            #     episode_length
            # ))

            # TODO save these

        # return total_times, total_rewards, reward_mses


def calc_discounted_rewards(rewards, total_time):
    """rewards are reward at that specific timestep plus discounted value of future rewards in that trajectory"""
    # print("r", rewards)
    discounted_rewards = np.zeros_like(rewards)
    running_rewards = 0.
    for i in range(total_time - 1, -1, -1):  # step backwards summary_varsin time from the end of the episode
        discounted_rewards[i] = rewards[i] + DISCOUNT_FACTOR * running_rewards
        running_rewards += discounted_rewards[i]
    # feature scaling/normalizing using standardization. reward vec will always have 0 mean and variance 1
    # otherwise, early rewards become astronomical as the running reward gets added to each previous ts
    # scaled rewards are much smaller, which makes it more stable for the neural network to approximate.
    # since we subtract the mean, we also see that the lat
    discounted_rewards -= discounted_rewards.mean()
    # decided to not subtract mean because the second half of the episode will never have a positive score.
    # that isn't necessarily good because sometimes we simply run out of time (max episode length is only 200).
    # we don't want to always be penalizing the end of an episode
    discounted_rewards /= discounted_rewards.std()
    # print("r", discounted_rewards)
    return discounted_rewards


def build_summaries():
    episode_time = tf.Variable(0.)
    a = tf.summary.scalar("Episode length", episode_time)
    episode_sum_discounted_reward = tf.Variable(0.)
    b = tf.summary.scalar("Total discounted reward", episode_sum_discounted_reward)
    episode_avg_reward = tf.Variable(0.)
    c = tf.summary.scalar("Average Reward per action", episode_avg_reward)

    summary_vars = [episode_time, episode_sum_discounted_reward, episode_avg_reward]
    summary_ops = tf.summary.merge([a,b,c])

    return summary_ops, summary_vars


# def plot_metadata(total_times, total_rewards, reward_mses):
#     f, axarr = plt.subplots(3, sharex=True) #sharex [True] must be one of ['all', 'row', 'col', 'none']
#     x = np.arange(N_EPISODES)
#     axarr[0].plot(x, total_times)
#     axarr[0].set_title('Total Times')
#     axarr[0].set_ylabel("Time")
#     axarr[1].plot(x, total_rewards)
#     axarr[1].set_title('Total Rewards')
#     axarr[1].set_ylabel("Rewards")
#     axarr[2].plot(x, reward_mses)
#     axarr[2].set_ylabel("Critic Weward MSE")
#     axarr[2].set_xlabel("Episode Number")
#
#     plt.show()


def main(_):
    for learning_rate in [1E-4, 1e-3, 1e-2]:
        for use_two_fc in [False]:
            for n_neurons in [20,50,80]:
                for use_dropout in [False]:
                    # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                    hparam = make_hparam_string(learning_rate, n_neurons, use_two_fc, use_dropout)
                    print('Starting run for %s' % hparam)
                    train(learning_rate, n_neurons, use_two_fc, use_dropout, hparam)
        # total_times, total_rewards, reward_mses = train(sess, env, policy)
        # plot_metadata(total_times, total_rewards, reward_mses)

        # TODO save progress to resume learning weights

        # gym.upload('/tmp/cartpole-experiment-1', api_key='ZZZ')


if __name__ == '__main__':
    tf.app.run()
