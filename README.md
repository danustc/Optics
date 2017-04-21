# Optics  
This is a small package of microscopy data processing. Can be used in combination with InControl package or for data analysis.  
Author: [danustc](https://github.com/danustc)
* **
## Package Instruction by Folders
###  PR(Phase Retrieval)
Contains codes for phase retrieval. Methods based on Hanser _et al._, "Phase-retrieved pupil functions in wide-field fluorescence microscopy", **J. Microscopy**, 216, 32--48, 2004.    

**Oblique_aberration.py**: the main program for processing _.mat_ data sets of the oblique SPIM. All the retrieved pupil functions are stored in a dictionary and saved as an _.npz_ file.

* **group_retrieval**: (kinda) the main program that processes all the _.mat_ files in a selected folder. Returns a list of Strehl ratios and retrieved pupils. The core algorithm is contained in the class **PSF_PF**, which must take the microscope parameters upon initialization.

* **load_mat**: load a _.mat_ file (file name _mat_path_) and convert it into an numpy array. Notice that 3D arrays in matlab and python have different orders of dimensions, which requires a transposition in one of them to match the two.

**Phase_retrieval.py**: Adapted from the phase-retrieval codes in Ryan's [InControl](https://github.com/Huanglab-ucsf/InControl) package.

* **PSF_PF**: the class of phase-retrieval. Users should provide a PSF (3-D numpy array), pixel size, z-step size, fluorescence wavelength (all in microns), refractive index (by default, water), numerical aperture (by default 1.0), focal length of objective **(_f_(tube lens)/Manification)** and number of iterations. Once the PSF is loaded, the pupil function (complex numpy 2-D array) can be retrieved by calling the function **retrievePF** and result would be saved in the class member PF. The unwrapped phase part can be acquired by calling the function **get_phase**.

* **_PupilFunction**: A private class tracking the properties of a pupil function.

**psf_tools.py**: some mini tools for slicing/visualizing PSF stacks.

**pupil.py**: called by **retrievePF** in **Phase_retrieval.py**. The core of the core :).
*  **Pupil**: class that defines a pupil function. Contains a bunch of redundant functions and properties which are useless but harmless.
* **Simulation**: the class containing the algorithms for conversions between a psf and its associated pupil function.

**visualization.py**: functions that plots pupil functions and zernike components.


### src (raw codes) ---- to be filled up
