# simulate the deformable mirror 
# No calculation for the pupil function
import numpy as np 
import libtim.zern 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy import optimize, integrate
import pupil2device as pupil
from numpy.lib.scimath import sqrt as _msqrt
from Phase_retrieval import DM_simulate
import time



        

def main():
    ndeg = 22
    inner_prod = np.zeros([ndeg,ndeg])
    DM = DM_simulate(nseg = 256, nPixels = 256)
    for N1 in np.arange(ndeg):
        zen_1 = DM.zernike_single(N1+1, 1.0)
#         seg_1 = DM.findSegOffsets(zen_1)
        seg_1 = DM.findSeg()
        seg_1 = seg_1.ravel()
        for N2 in np.arange(N1+1):
            zen_2 = DM.zernike_single(N2+1, 1.0)
            seg_2=DM.findSeg()
#             seg_2 = DM.findSegOffsets(zen_2)
            seg_2 = seg_2.ravel()
            inner_prod[N1, N2] = np.inner(seg_1, seg_2) / len(seg_1)
            inner_prod[N2, N1] = np.inner(seg_1, seg_2) / len(seg_1)#     pylab.imshow(seg_1)
            print(N1,N2)
            
    im = plt.imshow(inner_prod,interpolation='none',cmap = 'RdBu')
    plt.colorbar(im)
    plt.savefig('seg_256_new')
#     plt.imshow()
#     print(inner_prod)


if __name__ == "__main__":
    stime = time.time()
    main()
    etime = time.time()-stime
    print(etime)
# print(zen_1.shape)
# pylab.imshow(zen_1)
# pylab.show()
