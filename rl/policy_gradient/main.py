"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI
@project: 
@description:
"""

import numpy as np
import tensorflow as tf
import gym

def discounted_future_value(rewards, gamma=0.99):
  nr_episodes = rewards.shape[0]
  lst_discounted_reward = []

  for i in range(nr_episodes):
    discount_factor_per_step = np.array([gamma**(j-i) for j in range(i, nr_episodes)])
    lst_discounted_reward.append(discount_factor_per_step.dot(rewards[i:]))

  return np.array(lst_discounted_reward)  


class Agent:
  
  def __init__(self, input_size, output_size):
    self.input_size = input_size
    self.output_size = output_size
    self.model = None
    
  def __call__(self):
    tf_inp = tf.keras.layers.Input((self.input_size,), name='inp_state')
  
    tf_x = tf_inp
  
    for n_units in [16]:
      tf_x = tf.keras.layers.Dense(n_units, activation='relu')(tf_x)
    
    tf_out = tf.keras.layers.Dense(self.output_size, activation='softmax')(tf_x)
    
    self.model = tf.keras.Model(inputs=tf_inp, outputs=tf_out)
    self.model.compile(loss='categorical_crossentropy', optimizer='sgd')
    print("Model created: \n{}".format(self.model.summary()))
    return
  
  def get_action(self, env, episode, np_state=None):
    
    p_randomize = 1 - min(1,episode / 30)
    randomize = bool(np.random.choice([0,1], p=[1-p_randomize, p_randomize]))

    if self.model is None or randomize:
      return env.action_space.sample()
    
    if len(np_state.shape) == 1:
      np_state = np.expand_dims(np_state, 0)

    softmax = self.model.predict(np_state).ravel()
    return np.random.choice(np.arange(len(softmax)), p=softmax)
    
  def train_on_episode(self, states, rewards, actions):
    discounted_future_values = discounted_future_value(rewards, GAMMA)
    
    categ_actions = np.zeros((len(actions), self.output_size))
    categ_actions[np.arange(len(actions)), actions] = 1
    
    y_true = discounted_future_values.reshape((-1,1)) * categ_actions
    print("Training agent on {} observations".format(len(states)))
    loss = self.model.train_on_batch(states, y_true)
    return loss
  
    


if __name__ == '__main__':
  ENV_NAME = 'CartPole-v0'
  GAMMA = 0.99
  print("DBG", flush=True)
  env = gym.make(ENV_NAME)
  input_size = env.observation_space.shape[0]
  output_size = env.action_space.n
  agent = Agent(input_size=input_size, output_size=output_size)
  agent()

  #### TRAINING PHASE ######
  n_episodes = 100
  n_steps_per_ep = 1000
  
  print("Training for {} episodes (n steps per episodes = {})"
        .format(n_episodes, n_steps_per_ep))
  for ep in range(100):
    init_state = env.reset()
    states = [init_state]
    rewards = []
    actions = []
    crt_state = init_state
    for t in range(1000):
      # env.render()
      act = agent.get_action(env, ep, crt_state)
      state, reward, done, info = env.step(act)
      actions.append(act)
      rewards.append(reward)
      
      if done:
        print(" Ep {} - Reached point of no return".format(ep+1), flush=True)
        break
      
      states.append(state)
      crt_state = state
    #endfor
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    ep_loss = agent.train_on_episode(states, rewards, actions)
    print("Ep {} - finished - ep loss = {:.2f}".format(ep+1, ep_loss), flush=True)

  ### EVALUATION PHASE #####
  crt_state = env.reset()
  done = False
  n_iterations = 0
  while not done:
    env.render()
    act = agent.get_action(env, np.inf, crt_state)
    crt_state, reward, done, info = env.step(act)
    n_iterations+1
    
    if done:
      print("Evaluate phase - Reached point of no return after {} iters".format(n_iterations), flush=True)
      break
  
  env.close()