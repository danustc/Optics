# -*- coding: utf-8 -*-
#
# Copyright © 2009 Pierre Raybaut
# Licensed under the terms of the MIT License

"""
MatplotlibWidget
================

Example of matplotlib widget for PyQt4 and PyQt5

Copyright © 2009 Pierre Raybaut
This software is licensed under the terms of the MIT License

Derived from 'embedding_in_pyqt4.py':
Copyright © 2005 Florent Rougon, 2006 Darren Dale

Updated for PyQt5 compatibility by Jérémy Goutin, 2015
"""
__version__ = "1.1.0"

import sys

# =============================================================================
#   Example - Part 1 : Initialize PyQt (By importing Widjets for Part 2)
#                      Initialize Matplotlib for PyQt5 if needed
# =============================================================================
if __name__ == '__main__':
    try:
        # Use PyQt5 if present
        from PyQt5.QtWidgets import QMainWindow, QApplication
        import matplotlib
        matplotlib.use("Qt5Agg")
        print("Running Example with PyQt5...")
    except:
        # Else, use PyQt4
        from PyQt4.QtGui import QMainWindow, QApplication
        print("Running Example with PyQt4...")
# =============================================================================
#   End Example - Part 1
# =============================================================================

# Import PyQt Widgets and Matplotlib canvas for actually used PyQt version
if "PyQt5" in sys.modules:
    from PyQt5.QtWidgets import QSizePolicy
    from PyQt5.QtCore import QSize
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
elif "PyQt4" in sys.modules:
    from PyQt4.QtGui import QSizePolicy
    from PyQt4.QtCore import QSize
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
else:
    raise SystemError("PyQt4 or PyQt5 need to be imported first")

from matplotlib.figure import Figure

from matplotlib import rcParams
rcParams['font.size'] = 9


class MatplotlibWidget(Canvas):
    """
    MatplotlibWidget inherits PyQt4.QtGui.QWidget or PyQt5.QtWidgets.QWidget
    and matplotlib.backend_bases.FigureCanvasBase

    Options: option_name (default_value)
    -------
    parent (None): parent widget
    title (''): figure title
    xlabel (''): X-axis label
    ylabel (''): Y-axis label
    xlim (None): X-axis limits ([min, max])
    ylim (None): Y-axis limits ([min, max])
    xscale ('linear'): X-axis scale
    yscale ('linear'): Y-axis scale
    width (4): width in inches
    height (3): height in inches
    dpi (100): resolution in dpi
    hold (False): if False, figure will be cleared each time plot is called

    Widget attributes:
    -----------------
    figure: instance of matplotlib.figure.Figure
    axes: figure axes

    Example:
    -------
    self.widget = MatplotlibWidget(self, yscale='log', hold=True)
    from numpy import linspace
    x = linspace(-10, 10)
    self.widget.axes.plot(x, x**2)
    self.wdiget.axes.plot(x, x**3)
    """
    def __init__(self, parent=None, title='', xlabel='', ylabel='',
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 width=4, height=3, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if xscale is not None:
            self.axes.set_xscale(xscale)
        if yscale is not None:
            self.axes.set_yscale(yscale)
        if xlim is not None:
            self.axes.set_xlim(*xlim)
        if ylim is not None:
            self.axes.set_ylim(*ylim)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QSizePolicy.Expanding,
                             QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)

# =============================================================================
#   Example - Part 2 : Import MatplotlibWidget in PyQt Application
# =============================================================================
if __name__ == '__main__':
    from numpy import linspace

    class ApplicationWindow(QMainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.mplwidget = MatplotlibWidget(self, title='Example',
                                              xlabel='Linear scale',
                                              ylabel='Log scale',
                                              yscale='log')
            self.mplwidget.setFocus()
            self.setCentralWidget(self.mplwidget)
            self.plot(self.mplwidget.axes)

        def plot(self, axes):
            x = linspace(-10, 10)
            axes.plot(x, x**2)
            axes.plot(x, x**3)

    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
# =============================================================================
#   End Example - Part 2
# =============================================================================
