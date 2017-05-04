'''
This is the ui interface for phase retrieval.
Created by Dan on 05/04/2017 (Happy Youth's Day!)
'''

from PyQt4 import QtGui, QtCore
import UI_interface
import numpy as np
import libtim.zern as zern



class UI(UI_interface._UI):
    '''
    Update log and brief instructions.
    '''
    def __init__(self,  design_path):
        '''
        initialize the UI.
        control: the core functions which the UI calls
        design_path: the UI design.
        '''
        UI_interface._UI.__init__(self,design_path)
        #self._core= core

        # The connection group of the buttons and texts
        self._ui.pushButton_retrieve.clicked.connect(self.retrievePF)

    def retrievePF(self):
        # retrieve PF from psf
        print("function connected!")


# ------------------------Test of the module------------
def main():
    design_path = 'PR_design'
    PR_UI = UI(design_path)
    

if __name__ == '__main__':
    main()
