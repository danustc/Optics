"""
Last modification: 08/11/2016 by Dan.
# simulate the deformable mirror 
# No calculation for the pupil function

"""

import numpy as np 
import libtim.zern 
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation

class Zernike_func(object):
    def __init__(self,radius, mask=False):
        self.radius = radius
        self.useMask = mask
        self.pattern = []
        
    
    def single_zern(self, mode, amp):
        modes = np.zeros((mode))
        modes[mode-1] = amp
        self.pattern= libtim.zern.calc_zernike(modes, self.radius, mask = self.useMask, zern_data= {})
        
        return self.pattern
    
    
    def multi_zern(self, amps):
        self.pattern = libtim.zern.calc_zernike(amps, self.radius, mask = self.useMask, zern_data = {})
        return self.pattern
#         RMSE[ii] = np.sqrt(np.var(PF_core))
#         ii +=1

    def plot_zern(self):
        plt.imshow(self.pattern, interpolation='none')
        plt.show()
    

# ---------------Below is a simulation of deformable mirror
        
class DM_simulate(object):
    
    def __init__(self, nseg = 12, nPixels = 256, pattern=None):
        self.nSegments = nseg
        self.nPixels = nPixels
        self.DMsegs = np.zeros((self.nSegments, self.nSegments))
        self.zern = Zernike_func(nPixels/2)
        self.borders = np.linspace(0,self.nPixels,num=self.nSegments+1).astype(int)
        
        
        if pattern is None:
            self.pattern = np.zeros((nPixels,nPixels))
        else: 
            zoom = 256./np.float(pattern.shape[0])
            MOD = interpolation.zoom(pattern,zoom,order=0,mode='nearest')
            self.pattern = MOD
            

    def findSeg(self):
        for ii in np.arange(self.nSegments):
            for jj in np.arange(self.nSegments):
                xStart = self.borders[ii]
                xEnd = self.borders[ii+1]
                yStart = self.borders[jj]
                yEnd = self.borders[jj+1]
                
                av = np.mean(self.pattern[xStart:xEnd, yStart:yEnd])
                self.DMsegs[ii,jj] = av
                
        DMsegs = np.copy(self.DMsegs) # create a copy of self.segs
        return DMsegs
    
    
    def readSeg(self, raw_seg):
        """
        load a 1-d array and represent it with a segments
        return a matrix
        """ 
        seg=raw_seg[:140]
        seg = np.insert(seg,0,0)
        seg = np.insert(seg,11,0)
        seg = np.insert(seg,132,0)
        seg = np.insert(seg, 143, 0)
        rseg = np.flipud(seg.reshape(self.nSegments,self.nSegments).transpose())
        return rseg


    def zernSeg(self, zernmode): 
        """
        Display a zernike mode on the deformable mirror
        To be filled later.
        """
        self.pattern = self.zern.single_zern(zernmode, amp=1.)
        self.findSeg()
        
        # ------------------------------OK it's still good to add some visualizetion function. ------------------------
        