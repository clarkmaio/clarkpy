#from clarkpy.datasets.radio_plays import *
#from clarkpy.datasets.bolt_drive import *
#from clarkpy.datasets.uk_energy_market import *

from .radio_plays import load_radio_plays
from .taxi_drive import  load_taxi_drive
from .uk_energy_market import load_uk_gas_price
from .columbia_river import load_columbia_river

from urllib.request import urlopen
import pickle


def load_pickle_from_url(url: str):
    with urlopen(url=url) as f:
        d = pickle.load(f)
    return d