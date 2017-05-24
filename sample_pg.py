
import numpy as np
import tensorflow as tf
import os

# use correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use correct GPU

import gym
env = gym.make('CartPole-v0')

# env.reset()
# random_episodes = 0
# reward_sum = 0
# while random_episodes < 10:
#     env.render()
#     observation, reward, done, _ = env.step(np.random.randint(0,2))
#     reward_sum += reward
#     if done:
#         random_episodes += 1
#         print("Reward for this episode was:",reward_sum)
#         reward_sum = 0
#         env.reset()

# hyperparameters
N_NEURONS = 10 # number of hidden layer neurons
batch_size = 5 # every how many episodes to do a param update?
learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
gamma = 0.99 # discount factor for reward

STATE_DIM = 4 # input dimensionality

tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to
#giving a probability of chosing to the action of moving left or right.
input_states = tf.placeholder(tf.float32, [None, STATE_DIM], name="input_x")
W1 = tf.get_variable("W1", shape=[STATE_DIM, N_NEURONS],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(input_states, W1))
W2 = tf.get_variable("W2", shape=[N_NEURONS, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
print("score", np.shape(score))

probability = tf.nn.sigmoid(score)
print("p sha", np.shape(probability))

#From here we define the parts of the network needed for learning a good policy.
trainable_vars = tf.trainable_variables()
print("tvar shape", np.shape(trainable_vars))  # (2,)
print("tvar", trainable_vars)  # (2,)  [<tf.Variable 'W1:0' shape=(4, 10) dtype=float32_ref>, <tf.Variable 'W2:0' shape=(10, 1) dtype=float32_ref>]
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, trainable_vars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, trainable_vars))



def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


states, rewards, fake_labels, action_probs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()  # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(trainable_vars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        # if reward_sum / batch_size > 100 or rendering == True:
        #     env.render()
        #     rendering = True

        # Make sure the observation is in a shape the network can handle.
        state = np.reshape(observation, [1, STATE_DIM])

        # Run the policy network and get an action to take.
        action_prob = sess.run(probability, feed_dict={input_states: state})
        action = 1 if np.random.uniform() < action_prob else 0

        states.append(state)  # observation
        fake_label = 1 if action == 0 else 0  # a "fake label"
        fake_labels.append(fake_label)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        rewards.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            ep_states = np.vstack(states)
            ep_fake_labels = np.vstack(fake_labels)
            ep_rewards = np.vstack(rewards)
            ep_probabs = action_probs
            states, rewards, fake_labels, action_probs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_ep_rewards = discount_rewards(ep_rewards)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_ep_rewards -= np.mean(discounted_ep_rewards)
            discounted_ep_rewards /= np.std(discounted_ep_rewards)
            # print(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            gradients = sess.run(newGrads, feed_dict={input_states: ep_states, input_y: ep_fake_labels, advantages: discounted_ep_rewards})
            for ix, grad in enumerate(gradients):
                gradBuffer[ix] += grad  # gradients add onto themselves, variances smooth out

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(
                'Average reward for episode %f.  Total average reward %f.' % (
                reward_sum / batch_size, running_reward / batch_size))

                if reward_sum / batch_size > 200:
                    print(
                    "Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(
episode_number, 'Episodes completed.')