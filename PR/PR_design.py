# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PR_design.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Phase_retrieval(object):
    def setupUi(self, Phase_retrieval):
        Phase_retrieval.setObjectName(_fromUtf8("Phase_retrieval"))
        Phase_retrieval.resize(800, 600)
        self.centralwidget = QtGui.QWidget(Phase_retrieval)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(640, 20, 111, 61))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 90, 160, 421))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout_syspara = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_syspara.setObjectName(_fromUtf8("verticalLayout_syspara"))
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(200, 90, 551, 321))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout_display = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_display.setObjectName(_fromUtf8("gridLayout_display"))
        self.gview_psf = QtGui.QGraphicsView(self.gridLayoutWidget)
        self.gview_psf.setObjectName(_fromUtf8("gview_psf"))
        self.gridLayout_display.addWidget(self.gview_psf, 1, 0, 1, 1)
        self.horizontalScrollBar = QtGui.QScrollBar(self.gridLayoutWidget)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName(_fromUtf8("horizontalScrollBar"))
        self.gridLayout_display.addWidget(self.horizontalScrollBar, 2, 0, 1, 1)
        self.gview_pupil = QtGui.QGraphicsView(self.gridLayoutWidget)
        self.gview_pupil.setObjectName(_fromUtf8("gview_pupil"))
        self.gridLayout_display.addWidget(self.gview_pupil, 1, 1, 1, 1)
        self.label = QtGui.QLabel(self.gridLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_display.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_display.addWidget(self.label_2, 0, 1, 1, 1)
        self.gridLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(30, 20, 611, 61))
        self.gridLayoutWidget_2.setObjectName(_fromUtf8("gridLayoutWidget_2"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.pushButton_2 = QtGui.QPushButton(self.gridLayoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.lineEdit_2 = QtGui.QLineEdit(self.gridLayoutWidget_2)
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.textEdit = QtGui.QTextEdit(self.gridLayoutWidget_2)
        self.textEdit.setObjectName(_fromUtf8("textEdit"))
        self.gridLayout.addWidget(self.textEdit, 0, 2, 1, 1)
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(200, 420, 551, 91))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton_3 = QtGui.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.horizontalLayout.addWidget(self.pushButton_3)
        Phase_retrieval.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Phase_retrieval)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        Phase_retrieval.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Phase_retrieval)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Phase_retrieval.setStatusBar(self.statusbar)

        self.retranslateUi(Phase_retrieval)
        QtCore.QMetaObject.connectSlotsByName(Phase_retrieval)

    def retranslateUi(self, Phase_retrieval):
        Phase_retrieval.setWindowTitle(_translate("Phase_retrieval", "MainWindow", None))
        self.pushButton.setText(_translate("Phase_retrieval", "Retrieve!", None))
        self.label.setText(_translate("Phase_retrieval", "Measured psf", None))
        self.label_2.setText(_translate("Phase_retrieval", "Retrieved pupil", None))
        self.pushButton_2.setText(_translate("Phase_retrieval", "Load psf...", None))
        self.pushButton_3.setText(_translate("Phase_retrieval", "Zernike fit", None))

