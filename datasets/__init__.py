from radio_plays import *
from bolt_drive import *
from uk_energy_market import *
from urllib.request import urlopen
import pickle


def load_pickle_from_url(url: str):
    with urlopen(url=url) as f:
        d = pickle.load(f)
    return d

