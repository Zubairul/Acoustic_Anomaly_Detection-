
from os import makedirs
from os.path import sep
from os.path import expanduser
from os.path import isdir

def get_parameter(param):

    params = {
        # User specific paths (MODIFY THESE TO MATCH YOUR SETTINGS!!!)
        'root_path' : expanduser('~/repositories/Novia/Mirka'),
        'raw_eps_data_path' : expanduser('~/Novia data/Mirka/Ljudfiler'),
        'raw_data_path' : expanduser('~/ClassifierInterface/'),}

    if param in params.keys():
        return params[param]
