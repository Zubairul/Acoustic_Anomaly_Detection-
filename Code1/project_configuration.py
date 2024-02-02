
import numpy as np
from os import makedirs
from os.path import sep
from os.path import isdir
import Code1.my_configuration

def get_parameter(param, return_suggestions=False):

    params = {
        # User specific paths (modifyed in my_configuration.py)
        'root_path' : Code1.my_configuration.get_parameter('root_path'),
        'raw_eps_data_path' : Code1.my_configuration.get_parameter('raw_eps_data_path'),
        'raw_data_path' : Code1.my_configuration.get_parameter('raw_data_path'),

        # EPS signal segmentation parameters
        'moving_average_win_duration' : 0.1,  # s, the duration over which a moving arevage is to be calculated
        'subsample_interval' : 20,  # samples, subsample prior to labelling to speed up things
        'min_segment_duration' : 8,  # s, all segments shorter than this are removed
        'ramp_up_and_down_time' : 1, # s, ignore this many seconds fron the onset and offset of each segment

        # Newly recorded data at 10000 RPM Only
        'segment_duration' : 2,  # The length of a segment giving rise to one spectra
        'down_sample_rate' : 20,  # downsampling to make the spectrogram faster to calculate

        # Spectrum extraction parameters
        'time_warp' : True,  # Stretch time to match the RPM for all segments
        'frequency_resolution' : 10,  # Hz, frequency binning of the resulting spectrograms
        'min_real_rpm_samples' : 2**20,  # min samples to reach the desired bin resolution, segments are padded if needed
        'cutoff_frequency' : 6000,  # Hz, the highest frequency to include

        # Plot parameters
        'color_novia_red': np.array([159., 37., 26.]) / 255,
    }

    # Sub-directories to be created if not existing
    params['spectral_data_dir'] = params['root_path'] + sep + 'Spectral data' + sep

    # Return wanted parameter
    if param in params.keys():
        return params[param]
    # Return suggestions if asked for
    else:
        if return_suggestions:
            suggestions = [s for s in params.keys() if param in s]
            return suggestions

def set_up_project():
    # Get all sub-direcotries and create them if neeeded
    sub_dir_keys = get_parameter('dir', return_suggestions=True)
    for key in sub_dir_keys:
        sub_dir = get_parameter(key)
        if not isdir(sub_dir):
            makedirs(sub_dir)
            

def get_machine_info(tag):

    machine_info = {
    'o1' : (1, 'OK', 0, ''),
    'o2' : (2, 'OK', 0, ''),
    'o3' : (3, 'Faulty', 1, 'Uneven plate'),
    'o4' : (4, 'Faulty', 2, 'Fan problem'),
    'o5' : (5, 'Faulty', 2, 'Fan problem'),
    'o6' : (6, 'Faulty', 2, 'Fan problem'),
    'o7' : (7, 'Faulty', 3, 'Bearing problem'),
    'o8' : (8, 'Faulty', 3, 'Bearing problem'),
    'o9' : (9, 'Faulty', 3, 'Bearing problem'),
    'o10' : (10, 'Faulty', 4, 'Spider bearing'),
    'n1' : (11, 'OK', 0, 'Gen. 2 motor'),
    'n2' : (12, 'OK', 0, 'Gen. 2 motor'),
    'n3' : (13, 'OK', 0, 'Gen. 2 motor'),
    'n4' : (14, 'OK', 0, ''),
    'n5' : (15, 'OK', 0, ''),
    'n6' : (16, 'OK', 0, ''),
    'n7' : (17, 'OK', 0, ''),
    'n8' : (18, 'OK', 0, ''),
    'n9' : (19, 'OK', 0, ''),
    'n10' : (20, 'OK', 0, ''),
    'n11' : (21, 'OK', 0, ''),
    'n12' : (22, 'Faulty', 2, 'Fan problem'),
    'n13' : (23, 'Faulty', 5, 'Spindle bearing'),
    'n14' : (24, 'OK', 0, ''),
    'n15' : (25, 'OK', 0, ''),
    'n16' : (26, 'OK', 0, ''),
    'n17' : (27, 'OK', 0, ''),
    'n18' : (28, 'OK', 0, ''),
    'n19' : (29, 'OK', 0, ''),
    'n20' : (30, 'OK', 0, ''),
    'Z1' : (11, 'OK', 0, 'Gen. 2 motor'),
    'Z2' : (12, 'OK', 0, 'Gen. 2 motor'),
    'Z3' : (13, 'OK', 0, 'Gen. 2 motor'),
    'Z4' : (14, 'OK', 0, ''),
    'Z5' : (15, 'OK', 0, ''),
    'Z6' : (16, 'OK', 0, ''),
    'Z07' : (17, 'OK', 0, ''),
    'Z8' : (18, 'OK', 0, ''),
    'Z9' : (19, 'OK', 0, ''),
    'Z10' : (20, 'OK', 0, ''),
    'Z11' : (21, 'OK', 0, ''),
    'Z12' : (22, 'Faulty', 2, 'Fan problem'),
    'Z13' : (23, 'Faulty', 5, 'Spindle bearing'),
    'Z14' : (24, 'OK', 0, ''),
    'Z15' : (25, 'OK', 0, ''),
    'Z16' : (26, 'OK', 0, ''),
    'Z17' : (27, 'OK', 0, ''),
    'Z18' : (28, 'OK', 0, ''),
    'Z19' : (29, 'OK', 0, ''),
    'Z20' : (30, 'OK', 0, '')
    }

    return machine_info[tag]
