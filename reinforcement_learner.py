import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import tflearn
import os

# use correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use correct GPU

ENV_NAME = 'CartPole-v0'
RENDER_ENV = False
SAVE_METADATA = True
SAVE_VIDS = True
VIDEO_DIR = './results/videos'
TENSORBOARD_RESULTS_DIR = './results/tensorboard'

STATE_DIM = 4
ACTION_DIM = 1
ACTION_PROB_DIMS = 2
ACTION_BOUND = 1  # 0 to 1
ACTION_SPACE = [0, 1]

N_EPISODES = 50000
MAX_EP_STEPS = 200  # from CartPole env
ACTOR_LEARNING_RATE = 0.0001  # 0.0001
CRITIC_LEARNING_RATE = 0.0001
# discount factor
# the relevant window into the future is about 10 timesteps. (1 *0.7)^10 shrinks to 3% after 10 timesteps.
DISCOUNT_FACTOR = 0.7  # aka gamma

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

        self.actor = ActorNetwork(self.sess)
        self.critic = CriticNetwork(self.sess, self.actor.n_trainable_params)

    def calc_action_probabilities(self, observed_states):
        action_probabilities = self.sess.run(self.actor.action_predictor,
                                             feed_dict={self.actor.input_states: observed_states})
        action_probabilities = action_probabilities[0]  # reduce depth by 1
        # print("tf action probs", action_probabilities)
        return action_probabilities

    def predict_rewards(self, observed_states, observed_actions):
        """in case we want to peek at our reward predictions"""
        predicted_rewards = self.sess.run(self.critic.reward_predictor, feed_dict={
            self.critic.input_states: observed_states,
            self.critic.input_actions: observed_actions
        })
        return predicted_rewards

    def update_policy(self, observed_states, observed_actions, observed_rewards):
        """when episode concludes, lets update our actors and critics"""
        # print(np.shape(observed_states), np.shape(observed_actions), np.shape(observed_rewards))
        self.critic.optimize_critic_network(observed_states, observed_actions, observed_rewards)
        gradient_wrt_actions = self.critic.calc_action_gradient(observed_states, observed_actions)
        self.actor.optimize_actor_net(observed_states, gradient_wrt_actions)


class ActorNetwork(object):
    def __init__(self, sess):
        self.sess = sess
        self.n_units = 20
        self.input_states, self.action_predictor = self.mk_action_predictor_net()
        self.trainable_net_params = tf.trainable_variables()
        self.n_trainable_params = len(self.trainable_net_params)

        self.gradient_wrt_actions, self.actor_optimizer = self.mk_actor_optimizer()

    def mk_action_predictor_net(self):
        """neural network that outputs probabilities of each action"""
        input_states = tflearn.input_data(shape=[None, STATE_DIM], name='input_state')
        actor_net = tflearn.fully_connected(input_states, self.n_units, activation='relu',
                                            weights_init='truncated_normal',
                                            name='hidden1')
        actor_net = tflearn.dropout(actor_net, 0.5, name='actor_dropout')
        actor_net = tflearn.fully_connected(actor_net, ACTION_PROB_DIMS, activation='softmax',
                                            weights_init='truncated_normal', bias=True, bias_init='zeros',
                                            name='output_action_probabilities')

        return input_states, actor_net

    def mk_actor_optimizer(self):
        """optimizer for actor network"""
        # action gradient will be given to this by the critic network
        action_gradient_from_critic = tf.placeholder(tf.float32, [None, ACTION_DIM])
        # apply action gradient to network
        actor_gradients = tf.gradients(self.action_predictor, self.trainable_net_params, -action_gradient_from_critic)
        optimizer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE). \
            apply_gradients(zip(actor_gradients, self.trainable_net_params))
        return action_gradient_from_critic, optimizer

    def optimize_actor_net(self, in_states, action_gradient):
        """runs optimizer using gradient from critic network"""
        # print(np.shape(action_gradient))
        self.sess.run(self.actor_optimizer,
                      feed_dict={self.input_states: in_states, self.gradient_wrt_actions: action_gradient})


class CriticNetwork(object):
    def __init__(self, sess, n_actor_params):
        self.sess = sess
        self.n_units = 50

        self.input_states, self.input_actions, self.reward_predictor = self.mk_reward_predictor_network()
        # get full list of trainable params from tf, then slice out the ones belonging to the actor network
        self.trainable_params = tf.trainable_variables()[n_actor_params:]

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
        r_net = tflearn.dropout(r_net, 0.5, name='critic_dropout')
        r_net = tflearn.fully_connected(r_net, 1, weights_init='truncated_normal',
                                        bias=True, bias_init='zeros',
                                        name='output_rewards')

        return input_states, input_actions, r_net

    def mk_reward_network_optimizer(self):
        observed_rewards = tflearn.input_data(shape=[None, 1], name='input_rewards')
        loss = tflearn.mean_square(observed_rewards, self.reward_predictor)
        optimizer = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(loss)
        return observed_rewards, optimizer

    def optimize_critic_network(self, observed_states, observed_action,
                                observed_rewards):  # note: replaced predicted_q_value with sum of mixed rewards
        """update our predictions based on observations"""
        self.sess.run([self.reward_predictor, self.critic_optimizer], feed_dict={
            self.input_states: observed_states,
            self.input_actions: observed_action,
            self.observed_rewards: observed_rewards
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
        action_gradient = action_gradient[0]  # compensate for nested list
        return action_gradient


def train(sess, env, policy):
    sess.run(tf.global_variables_initializer())
    # Set up summary Ops
    # TODO use these
    summary_ops, summary_vars = build_summaries()

    total_times = []
    total_rewards = []
    reward_mses = []

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
            #print("reward", reward)
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
        #print(discounted_rewards)

        # make our env observations into correct tensor shapes
        tensor_lengths = len(actions)
        actions = np.reshape(actions, (tensor_lengths, 1))
        discounted_rewards = np.reshape(discounted_rewards, (tensor_lengths, 1))

        # update policy
        policy.update_policy(states, actions, discounted_rewards)

        total_rewards.append(discounted_rewards.sum())

        # let's look at how our reward belief network is doing
        reward_mse = np.mean((discounted_rewards - policy.predict_rewards(states, actions))**2)
        reward_mses.append(reward_mse)

        # print("episode {} | total reward {} | avg reward {} | time alive {} | reward loss {}".format(
        #     episode,
        #     discounted_rewards.sum(),
        #     discounted_rewards.mean(),
        #     total_time,
        #     reward_mse
        # ))

        # TODO save these

    return total_times, total_rewards, reward_mses



def calc_discounted_rewards(rewards, total_time):
    """rewards are reward at that specific timestep plus discounted value of future rewards in that trajectory"""
    # print("r", rewards)
    discounted_rewards = np.zeros_like(rewards)
    running_rewards = 0.
    for i in range(total_time - 1, -1, -1):  # step backwards in time from the end of the episode
        discounted_rewards[i] = rewards[i] + DISCOUNT_FACTOR * running_rewards
        running_rewards += discounted_rewards[i]
    # feature scaling/normalizing using standardization. reward vec will always have 0 mean and variance 1
    # otherwise, early rewards become astronomical as the running reward gets added to each previous ts
    # scaled rewards are much smaller, which makes it more stable for the neural network to approximate.
    # since we subtract the mean, we also see that the lat
    #discounted_rewards -= discounted_rewards.mean()
    # decided to not subtract mean because the second half of the episode will never have a positive score.
    # that isn't necessarily good because sometimes we simply run out of time (max episode length is only 200).
    # we don't want to always be penalizing the end of an episode
    discounted_rewards /= discounted_rewards.std()
    #print("r", discounted_rewards)
    return discounted_rewards


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def plot_metadata(total_times, total_rewards, reward_mses):
    f, axarr = plt.subplots(3, sharex=True) #sharex [True] must be one of ['all', 'row', 'col', 'none']
    x = np.arange(N_EPISODES)
    axarr[0].plot(x, total_times)
    axarr[0].set_title('Total Times')
    axarr[0].set_ylabel("Time")
    axarr[1].plot(x, total_rewards)
    axarr[1].set_title('Total Rewards')
    axarr[1].set_ylabel("Rewards")
    axarr[2].plot(x, reward_mses)
    axarr[2].set_ylabel("Critic Weward MSE")
    axarr[2].set_xlabel("Episode Number")

    plt.show()


def main(_):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(TENSORBOARD_RESULTS_DIR, sess.graph)
        env = gym.make(ENV_NAME)
        policies = {'random': RandomPolicy, 'contrarian': ContrarianPolicy, 'policy_gradient': PolicyGradient}
        policy = policies['policy_gradient'](sess)
        env = gym.wrappers.Monitor(env, VIDEO_DIR, force=True)
        total_times, total_rewards, reward_mses = train(sess, env, policy)
        plot_metadata(total_times, total_rewards, reward_mses)

        # TODO save progress to resume learning weights

        # TODO: figure out how to make tf tensorboard graphs

        # gym.upload('/tmp/cartpole-experiment-1', api_key='ZZZ')


if __name__ == '__main__':
    tf.app.run()
