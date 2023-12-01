import io
import base64
from IPython.display import HTML
import gym
import numpy as np
import cv2

def video(filename, width=None):
    encod = base64.b64encode(io.open(filename, 'r+b').read())
    video_width = 'width="' + str(width) + '"' if width is not None else ''
    embedd = HTML(data='''
        <video controls {0}>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>'''.format(video_width, encod.decode('ascii')))

    return embedd


def preprocess(image):
    I = image[35:195] 
    I = I[::2, ::2, 0] 
    I[I == 144] = 0 
    I[I == 109] = 0 
    I[I != 0] = 1 
    I = cv2.dilate(I, np.ones((3, 3), np.uint8), iterations=1)
    I = I[::2, ::2, np.newaxis]
    return I.astype(np.float)


def change(prev, curr):
    prev = preprocess(prev)
    curr = preprocess(curr)
    I = prev - curr
    return I


class Memory:
  def __init__(self): 
      self.clear()

  def clear(self): 
      self.observs = []
      self.actions = []
      self.rewards = []

  def add_to_memory(self, new_observ, new_action, new_reward): 
      self.observs.append(new_observ)
      self.actions.append(new_action)
      self.rewards.append(new_reward)

    
def aggregate_memories(memories):
  batch_memory = Memory()
  
  for memory in memories:
    for step in zip(memory.observations, memory.actions, memory.rewards):
      batch_memory.add_to_memory(*step)
  
  return batch_memory


def parallelized_rollouts(batch_size, envs, model, choose_action):

    assert len(envs) == batch_size, "Number of parallel environments must be equal to the batch size."

    memories = [Memory() for _ in range(batch_size)]
    next_observ = [single_env.reset() for single_env in envs]
    prev_frames = [obs for obs in next_observ]
    done = [False] * batch_size
    rewards = [0] * batch_size

    while True:

        curr_frames = [obs for obs in next_observ]
        diff_frames = [change(prev, curr) for (prev, curr) in zip(prev_frames, curr_frames)]

        diff_frames_not_done = [diff_frames[b] for b in range(batch_size) if not done[b]]
        actions_not_done = choose_action(model, np.array(diff_frames_not_done), single=False)

        actions = [None] * batch_size
        ind_not_done = 0
        for b in range(batch_size):
            if not done[b]:
                actions[b] = actions_not_done[ind_not_done]
                ind_not_done += 1

        for b in range(batch_size):
            if done[b]:
                continue
            next_observ[b], rewards[b], done[b], info = envs[b].step(actions[b])
            prev_frames[b] = curr_frames[b]
            memories[b].add_to_memory(diff_frames[b], actions[b], rewards[b])

        if all(done):
            break

    return memories


def save_video(model, env_name, suffix=""):
    import skvideo.io
    from pyvirtualdisplay import Display
    display_video = Display(visible=0, size=(400, 300))
    display_video.start()

    env = gym.make(env_name)
    obs = env.reset()
    prev_obs = obs

    filename = env_name + suffix + ".mp4"
    video_output = skvideo.io.FFmpegWriter(filename)

    counter = 0
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        video_output.writeFrame(frame)

        if "CartPole" in env_name:
            input_obs = obs
        elif "Pong" in env_name:
            input_obs = change(prev_obs, obs)
        else:
            raise ValueError(f"Unknown env for saving: {env_name}")

        action = model(np.expand_dims(input_obs, 0)).numpy().argmax()

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        counter += 1

    video_output.close()
    print("Successfully saved {} frames into {}!".format(counter, filename))
    return filename


def save_video_memory(memory, filename, size=(512,512)):
    import skvideo.io

    video_output = skvideo.io.FFmpegWriter(filename)

    for observation in memory.observs:
        video_output.writeFrame(cv2.resize(255*observation, size))
        
    video_output.close()
    return filename
