# the simplest telescope
import numpy as np
import matplotlib.pyplot as plt

def telescope(f1, f2, l1, d):
    '''
    f1: the focal length of the first lens
    f2: the focal length of the second lens 
    l1: the distance between the object and the first lens
    d: the distance between the two lenses
    '''
    d1 = f1*l1/(l1-f1)
    print(d1)
    d2 = d-d1
    print(d2)
    l2 = f2*d2/(d2-f2)
    print(l2)
    M = (l2/d2)*(d1/l1)
    print(M)
    return l2, M


def main():
    f1 = 50
    f2 = 50
    d = 166
    l1 = np.arange(51,100)+50
    l2, M = telescope(f1,f2,l1,d)
    # plot the l2 and magnification
    fig = plt.figure(figsize = (6,8))
    ax1= fig.add_subplot(211)
    ax1.plot(l1,l2)
    ax2 = fig.add_subplot(212)
    ax2.plot(l1,M)
    fig.savefig('telescope')
    print("done!")

# run the program 
if __name__ == '__main__':
    main()
    
