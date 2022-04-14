from operator import truediv
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    self.init_state = np.zeros((700*5000,1),dtype=np.uint8)
    self._state = np.zeros((700*5000,1),dtype=np.uint8)
    self._step = 0
    self.action_space = spaces.Box(
      # low=np.array([1, 1, 1, 1]), # x y w h
      # high=np.array([20, 400, 50, 100]), # mm scale
      low = 0,
      high = 1,
      shape=(4,),
      # dtype=np.uint8
    )
    # 4 dimension
    self.observation_space = spaces.Box(
      low=0,
      high=1,
      # shape=(700,5000,1),
      shape = (700*5000,),
      # 0.1 mm
      dtype=np.uint8
    )
  def step(self, action):
    xx = action[0]*19+1
    yy = action[1]*399+1
    ww = action[2]*49+1
    hh = action[3]*99+1
    x=np.around(xx*10)
    y=np.around(yy*10)
    w=np.around(ww*10)
    h=np.around(hh*10)
    temp_state = np.zeros((700,5000,1),dtype=np.uint8)
    temp_state[x:w,y:h] = 1
    temp_state2 = temp_state.reshape((700*5000,1))
    if np.sum(ob*temp_state2)==0:
      checker = True
    else:
      checker = False

    if checker==True:
      ob = self._state + temp_state
    else:
      ob = self._state
    rew = np.sum(ob)
    done = True if self._step > 6 else False
    info = {}
    self._state = ob
    self._step += 1
    return ob, rew, done, info
    # step reward terminal info: none
  def reset(self):
    return self.init_state
    # initial state
  def render(self, mode='human', close=False):
    temp_state = self._state.reshape(700,5000)
    return temp_state
    # imageio.imread

# def reward(action):
#   pass