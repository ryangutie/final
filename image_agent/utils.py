# https://docs.python.org/3/reference/index.html
import numpy as np
import pystk
from .model import PuckDetector
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms
from PIL import Image
from glob import glob
from os import path, makedirs
from torch import load
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# Constants
RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'data\\trainTest'

class SuperTuxDataset(Dataset):
    """ Dataset for loading and transforming Super Tux Kart image data. """
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png')).load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image, label)

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=4, batch_size=128):
    """ Loads data into a DataLoader for batch processing. """
    dataset = SuperTuxDataset(dataset_path, transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

class PyTux:
    """ Manages the Super Tux Kart game environment. """
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Singleton instance already created"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """ Simulates a round of gameplay, collecting data. """
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
        else:
            if self.k: self.k.stop(); del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, render=True, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            self.k = pystk.Race(config); self.k.start()
        self.k.step()
        state, track = pystk.WorldState(), pystk.Track()
        for t in range(max_frames):
            state.update(); track.update()
            if data_callback: data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)
            if np.linalg.norm(state.players[0].kart.velocity) < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t; action.rescue = True
            self.k.step(action)

    def close(self):
        """ Cleans up the game environment. """
        if self.k: self.k.stop(); del self.k
        pystk.clean()

def load_model():
    """ Loads the AI model. """
    r = PuckDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'model.th'), map_location='cpu'))
    return r

if __name__ == '__main__':
    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_images', default=10000, type=int)
    parser.add_argument('-m', '--steps_per_track', default=20000, type=int)
    parser.add_argument('--aim_noise', default=0.1, type=float)
    parser.add_argument('--vel_noise', default=5, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    try: makedirs(args.output)
    except OSError: pass

    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        while n < args.steps_per_track:
            steps, how_far = pytux.rollout(track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect)
            print(steps, how_far)
    pytux.close()
