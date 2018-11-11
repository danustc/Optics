'''
This is the ui interface for phase retrieval.
Created by Dan on 05/04/2017 (Happy Youth's Day!)
'''
import sys
from PyQt5 import QtWidgets, QtCore
import os


def import_module(path):
    filename = os.path.basename(path)
    return __import__(path, globals(), locals(), [filename], 0)



class _UI(object):
    '''
    Update log and brief instructions.
    '''
    def __init__(self,core,design_path):
        # initialize the UI.
        self._core = core
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QWidget()
        self._window.closeEvent = self.shutDown

        ui_module = import_module(design_path)
        self._ui = ui_module.Ui_Form()
        try:
            self._ui.setupUi(self._window)
            print("UI setup successful!")
        except:
            self._window = QtWidgets.QMainWindow()
            self._ui.setupUi(self._window)
            print("UI setup successful!")

        self._window.show()
        self._app.exec_()

