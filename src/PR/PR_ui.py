'''
This is the ui interface for phase retrieval.
Created by Dan on 05/04/2017 (Happy Youth's Day!)
'''

from PyQt4 import QtGui, QtCore
import UI_interface
import numpy as np
from PR_core import Core


class UI(UI_interface._UI):
    '''
    Update log and brief instructions.
    '''
    def __init__(self,  core, design_path):
        '''
        initialize the UI.
        core: the core functions which the UI calls
        design_path: the UI design.
        '''
        UI_interface._UI.__init__(self,design_path)
        self._core= core

        # The connection group of the buttons and texts
        self._ui.pushButton_retrieve.clicked.connect(self.retrievePF)
        self._ui.pushButton_loadpsf.clicked.connect(self.load_PSF)
        self._ui.lineEdit_NA.returnPressed.connect(self.set_NA)
        self._ui.lineEdit_nfrac.returnPressed.connect(self.set_nfrac)

    def load_PSF(self):
        '''
        load a psf function (.npy) from the selected folder
        '''
        filename = QtGui.QFileDialog.getOpenFilename(None, 'Open psf:', '', '*.npy')
        self._ui.lineEdit_loadpsf.setText(filename)
        self._core.load_psf(filename)



    def retrievePF(self):
        # retrieve PF from psf
        print("function connected!")
        mask_size = int(self._ui.lineEdit_mask.text())
        self.set_dz()
        self._core.retrievePF(self.dz, mask_size)


    # ------Below are a couple of setting functions ------
    def set_NA(self, NA_input = None):
        if NA_input is None:
            NA_input = float(self._ui.lineEdit_NA.text())
        self.NA = NA_input

    def set_nfrac(self, nfrac = None):
        if ncrac is None:
            nfrac = float(self._ui.lineEdit_nfrac.text())
        self._core.set_nfrac(nfrac)

    def set_dz(self, dz_input = None):
        if dz_input is None:
           dz_input = float(self._ui.lineEdit_dz.text())
        self.dz = dz # this is not stored in the core program.


    # ------Below are a couple of displaying functions ------------
    def display_psf(self, dimension = 0):
        '''
        display the psf across a plane.
        dimension = 0: lateral, 1: xz plane, 2: yz plane
        '''




        self._ui.mpl_psf.figure.axes[0].matshow(np.log(psf_slice, cmap = 'Greys_r'))
        self._ui.mpl_psf.draw()



# ------------------------Test of the module------------
def main():
    pr_core = Core()
    design_path = 'PR_design'
    PR_UI = UI(pr_core, design_path)
    

if __name__ == '__main__':
    main()
