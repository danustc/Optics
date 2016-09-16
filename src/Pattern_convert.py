"""
Pattern_convert: convert arbitrary intensity pattern on the imaging plane into phase on the pupil plane.
added by Dan on 09/16/2016. 
"""


import numpy as np
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt


class pt_convert(object):
    
    def __init__(self, pattern, objective):
        """
        load the pattern 
        """





