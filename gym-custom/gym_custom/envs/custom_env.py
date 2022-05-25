from operator import truediv
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    self.init_state = np.zeros((5*10),dtype=np.uint8)
    self._state = np.zeros((5*10),dtype=np.uint8)
    self._step = 0
    self.action_space = spaces.Box(
      low = 0,
      high = 1,
      shape=(4,),
    )
    # 4 dimension
    self.observation_space = spaces.Box(
      low=0,
      high=1,
      # shape=(700,5000,1),
      shape = (5*10,),
      # 1 mm
      dtype=np.uint8
    )
  def step(self, action):
    xx = action[0]*4
    yy = action[1]*9
    ww = action[2]*4
    hh = action[3]*9
    # print(xx,yy,ww,hh)
    x=int(np.around(xx))
    y=int(np.around(yy))
    w=int(np.around(ww))
    h=int(np.around(hh))
    # print('x:',x)
    # print('y:',y)
    # print('w:',w)
    # print('h:',h)
    # print('action:',x,y,w,h)
    temp_state = np.zeros((5,10,1),dtype=np.uint8)
    temp_state[x:w,y:h] = 1
    # print(np.sum(temp_state))
    temp_state2 = temp_state.reshape((5*10))
    if np.sum(self._state*temp_state2)==0:
      checker = True
    else:
      checker = False
    
    if checker==True:
      ob = self._state + temp_state2
    else:
      ob = self._state
      temp_state2 *= 0
    # print('obs:',ob)
    rew = np.sum(temp_state2)/50.
    # print('reward:',rew)
    done = True if self._step > 4 else False
    # import ipdb;ipdb.set_trace()
    info = {}
    self._state = ob
    self._step += 1
    return ob, rew, done, info
    # step reward terminal info: none
  def reset(self):
    self._step = 0
    self._state = self.init_state.copy()
    return self.init_state
  def render(self, mode='human', close=False):
    return self._state.reshape(5,10)

# def reward(action):
#   pass