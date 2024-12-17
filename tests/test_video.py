import gym
import isaacgym
import isaacgymenvs
import torch

"""
NOTE: if capture video does not work, sudo apt install xvfb ffmpeg

To solve this error:
  File "/workspace/isaac/IsaacGymEnvs/.pixi/envs/default/lib/python3.8/site-packages/PIL/ImageGrab.py", line 63, in grab
    raise OSError("Pillow was built without XCB support")
OSError: Pillow was built without XCB support

Do manual installation:

$ sudo apt install libxcb1-dev
$ pip install --no-cache-dir Pillow

To test:

from PIL import features
print(features.check_feature('xcb'))

"""

num_envs = 64

envs = isaacgymenvs.make(
	seed=0, 
	task="Ant", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=True,
	force_render=False,
)
envs.is_vector_env = True
envs = gym.wrappers.RecordVideo(
	envs,
	"./videos",
	step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
	video_length=100  # for each video record up to 100 steps
)
envs.reset()
print("the image of Isaac Gym viewer is an array of shape", envs.render(mode="rgb_array").shape)
for _ in range(100):
	actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	envs.step(actions)