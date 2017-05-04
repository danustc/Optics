'''
This is the ui interface for phase retrieval.
Created by Dan on 05/04/2017 (Happy Youth's Day!)
'''
import sys
from PyQt4 import QtGui, QtCore
import os


def import_module(path):
    filename = os.path.basename(path)
    return __import__(path, globals(), locals(), [filename], 0)



class _UI(object):
    '''
    Update log and brief instructions.
    '''
    def __init__(self,design_path):
        # initialize the UI.
        self._app = QtGui.QApplication(sys.argv)
        self._window = QtGui.QWidget()
        self._window.closeEvent = self.shutDown

        ui_module = import_module(design_path)
        self._ui = ui_module.Ui_Form()
        try:
            self._ui.setupUi(self._window)
        except:
            self._window = QtGui.QMainWindow()
            self._ui.setupUi(self._window)

        self._window.show()
        self._app.exec_()

    def shutDown(self, event):
        '''
        shut down the UI
        '''
        self._app.quit()
