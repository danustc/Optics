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
        Phase_retrieval.resize(1053, 700)
        self.centralwidget = QtGui.QWidget(Phase_retrieval)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton_retrieve = QtGui.QPushButton(self.centralwidget)
        self.pushButton_retrieve.setGeometry(QtCore.QRect(590, 20, 111, 61))
        self.pushButton_retrieve.setObjectName(_fromUtf8("pushButton_retrieve"))
        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 90, 160, 431))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout_syspara = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_syspara.setObjectName(_fromUtf8("verticalLayout_syspara"))
        self.label_NA = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_NA.setObjectName(_fromUtf8("label_NA"))
        self.verticalLayout_syspara.addWidget(self.label_NA)
        self.lineEdit_NA = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_NA.setObjectName(_fromUtf8("lineEdit_NA"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_NA)
        self.label_nfrac = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_nfrac.setObjectName(_fromUtf8("label_nfrac"))
        self.verticalLayout_syspara.addWidget(self.label_nfrac)
        self.lineEdit_nfrac = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_nfrac.setObjectName(_fromUtf8("lineEdit_nfrac"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_nfrac)
        self.label_objfl = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_objfl.setObjectName(_fromUtf8("label_objfl"))
        self.verticalLayout_syspara.addWidget(self.label_objfl)
        self.lineEdit_objfl = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_objfl.setObjectName(_fromUtf8("lineEdit_objfl"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_objfl)
        self.label_wlc = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_wlc.setObjectName(_fromUtf8("label_wlc"))
        self.verticalLayout_syspara.addWidget(self.label_wlc)
        self.lineEdit_wlc = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_wlc.setObjectName(_fromUtf8("lineEdit_wlc"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_wlc)
        self.label_nwl = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_nwl.setObjectName(_fromUtf8("label_nwl"))
        self.verticalLayout_syspara.addWidget(self.label_nwl)
        self.lineEdit_nwl = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_nwl.setObjectName(_fromUtf8("lineEdit_nwl"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_nwl)
        self.label_4 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_syspara.addWidget(self.label_4)
        self.lineEdit_3 = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_3)
        self.label_pxl = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_pxl.setObjectName(_fromUtf8("label_pxl"))
        self.verticalLayout_syspara.addWidget(self.label_pxl)
        self.lineEdit_pxl = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_pxl.setObjectName(_fromUtf8("lineEdit_pxl"))
        self.verticalLayout_syspara.addWidget(self.lineEdit_pxl)
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(200, 90, 745, 421))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout_display = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_display.setObjectName(_fromUtf8("gridLayout_display"))
        self.label_psfview = QtGui.QLabel(self.gridLayoutWidget)
        self.label_psfview.setObjectName(_fromUtf8("label_psfview"))
        self.gridLayout_display.addWidget(self.label_psfview, 0, 0, 1, 1)
        self.label_pupilview = QtGui.QLabel(self.gridLayoutWidget)
        self.label_pupilview.setObjectName(_fromUtf8("label_pupilview"))
        self.gridLayout_display.addWidget(self.label_pupilview, 0, 1, 1, 1)
        self.graphicsView_psf = QtGui.QGraphicsView(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_psf.sizePolicy().hasHeightForWidth())
        self.graphicsView_psf.setSizePolicy(sizePolicy)
        self.graphicsView_psf.setObjectName(_fromUtf8("graphicsView_psf"))
        self.gridLayout_display.addWidget(self.graphicsView_psf, 1, 0, 1, 1)
        self.graphicsView_pupil = QtGui.QGraphicsView(self.gridLayoutWidget)
        self.graphicsView_pupil.setObjectName(_fromUtf8("graphicsView_pupil"))
        self.gridLayout_display.addWidget(self.graphicsView_pupil, 1, 1, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton_yzview = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_yzview.sizePolicy().hasHeightForWidth())
        self.pushButton_yzview.setSizePolicy(sizePolicy)
        self.pushButton_yzview.setObjectName(_fromUtf8("pushButton_yzview"))
        self.horizontalLayout.addWidget(self.pushButton_yzview)
        self.pushButton_xyview = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_xyview.sizePolicy().hasHeightForWidth())
        self.pushButton_xyview.setSizePolicy(sizePolicy)
        self.pushButton_xyview.setObjectName(_fromUtf8("pushButton_xyview"))
        self.horizontalLayout.addWidget(self.pushButton_xyview)
        self.pushButton_xzview = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_xzview.sizePolicy().hasHeightForWidth())
        self.pushButton_xzview.setSizePolicy(sizePolicy)
        self.pushButton_xzview.setObjectName(_fromUtf8("pushButton_xzview"))
        self.horizontalLayout.addWidget(self.pushButton_xzview)
        self.pushButton_lineview = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_lineview.sizePolicy().hasHeightForWidth())
        self.pushButton_lineview.setSizePolicy(sizePolicy)
        self.pushButton_lineview.setObjectName(_fromUtf8("pushButton_lineview"))
        self.horizontalLayout.addWidget(self.pushButton_lineview)
        self.gridLayout_display.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.checkBox_rm4 = QtGui.QCheckBox(self.gridLayoutWidget)
        self.checkBox_rm4.setObjectName(_fromUtf8("checkBox_rm4"))
        self.horizontalLayout_3.addWidget(self.checkBox_rm4)
        self.pushButton_pfraw = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_pfraw.sizePolicy().hasHeightForWidth())
        self.pushButton_pfraw.setSizePolicy(sizePolicy)
        self.pushButton_pfraw.setObjectName(_fromUtf8("pushButton_pfraw"))
        self.horizontalLayout_3.addWidget(self.pushButton_pfraw)
        self.pushButton_pffit = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_pffit.sizePolicy().hasHeightForWidth())
        self.pushButton_pffit.setSizePolicy(sizePolicy)
        self.pushButton_pffit.setObjectName(_fromUtf8("pushButton_pffit"))
        self.horizontalLayout_3.addWidget(self.pushButton_pffit)
        self.pushButton_zbar = QtGui.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_zbar.sizePolicy().hasHeightForWidth())
        self.pushButton_zbar.setSizePolicy(sizePolicy)
        self.pushButton_zbar.setObjectName(_fromUtf8("pushButton_zbar"))
        self.horizontalLayout_3.addWidget(self.pushButton_zbar)
        self.horizontalLayout_2.addLayout(self.horizontalLayout_3)
        self.gridLayout_display.addLayout(self.horizontalLayout_2, 2, 1, 1, 1)
        self.gridLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(30, 20, 551, 62))
        self.gridLayoutWidget_2.setObjectName(_fromUtf8("gridLayoutWidget_2"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.lineEdit_2 = QtGui.QLineEdit(self.gridLayoutWidget_2)
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.pushButton_loadpsf = QtGui.QPushButton(self.gridLayoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_loadpsf.sizePolicy().hasHeightForWidth())
        self.pushButton_loadpsf.setSizePolicy(sizePolicy)
        self.pushButton_loadpsf.setObjectName(_fromUtf8("pushButton_loadpsf"))
        self.gridLayout.addWidget(self.pushButton_loadpsf, 0, 1, 1, 1)
        self.lineEdit_4 = QtGui.QLineEdit(self.gridLayoutWidget_2)
        self.lineEdit_4.setObjectName(_fromUtf8("lineEdit_4"))
        self.gridLayout.addWidget(self.lineEdit_4, 0, 2, 1, 1)
        self.horizontalLayoutWidget_4 = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(200, 520, 537, 61))
        self.horizontalLayoutWidget_4.setObjectName(_fromUtf8("horizontalLayoutWidget_4"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_pupilfname = QtGui.QLabel(self.horizontalLayoutWidget_4)
        self.label_pupilfname.setObjectName(_fromUtf8("label_pupilfname"))
        self.verticalLayout.addWidget(self.label_pupilfname)
        self.textEdit_pfname = QtGui.QTextEdit(self.horizontalLayoutWidget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit_pfname.sizePolicy().hasHeightForWidth())
        self.textEdit_pfname.setSizePolicy(sizePolicy)
        self.textEdit_pfname.setObjectName(_fromUtf8("textEdit_pfname"))
        self.verticalLayout.addWidget(self.textEdit_pfname)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.pushButton_pfbrowse = QtGui.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_pfbrowse.sizePolicy().hasHeightForWidth())
        self.pushButton_pfbrowse.setSizePolicy(sizePolicy)
        self.pushButton_pfbrowse.setObjectName(_fromUtf8("pushButton_pfbrowse"))
        self.horizontalLayout_4.addWidget(self.pushButton_pfbrowse)
        self.pushButton_savepupil = QtGui.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_savepupil.sizePolicy().hasHeightForWidth())
        self.pushButton_savepupil.setSizePolicy(sizePolicy)
        self.pushButton_savepupil.setObjectName(_fromUtf8("pushButton_savepupil"))
        self.horizontalLayout_4.addWidget(self.pushButton_savepupil)
        self.pushButton_savefit = QtGui.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_savefit.sizePolicy().hasHeightForWidth())
        self.pushButton_savefit.setSizePolicy(sizePolicy)
        self.pushButton_savefit.setObjectName(_fromUtf8("pushButton_savefit"))
        self.horizontalLayout_4.addWidget(self.pushButton_savefit)
        Phase_retrieval.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Phase_retrieval)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1053, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        Phase_retrieval.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Phase_retrieval)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Phase_retrieval.setStatusBar(self.statusbar)

        self.retranslateUi(Phase_retrieval)
        QtCore.QMetaObject.connectSlotsByName(Phase_retrieval)

    def retranslateUi(self, Phase_retrieval):
        Phase_retrieval.setWindowTitle(_translate("Phase_retrieval", "Phase Retrieval", None))
        self.pushButton_retrieve.setText(_translate("Phase_retrieval", "Retrieve!", None))
        self.label_NA.setText(_translate("Phase_retrieval", "Numerical aperture", None))
        self.lineEdit_NA.setText(_translate("Phase_retrieval", "1.0", None))
        self.label_nfrac.setText(_translate("Phase_retrieval", "Refractive index", None))
        self.lineEdit_nfrac.setText(_translate("Phase_retrieval", "1.33", None))
        self.label_objfl.setText(_translate("Phase_retrieval", "Focal length (mm)", None))
        self.lineEdit_objfl.setText(_translate("Phase_retrieval", "9.0", None))
        self.label_wlc.setText(_translate("Phase_retrieval", "Wavelength (nm)", None))
        self.lineEdit_wlc.setText(_translate("Phase_retrieval", "515.0", None))
        self.label_nwl.setText(_translate("Phase_retrieval", "# wavelengths", None))
        self.lineEdit_nwl.setText(_translate("Phase_retrieval", "3", None))
        self.label_4.setText(_translate("Phase_retrieval", "Wavelength steps (nm)", None))
        self.lineEdit_3.setText(_translate("Phase_retrieval", "5.0", None))
        self.label_pxl.setText(_translate("Phase_retrieval", "Pixel size (nm)", None))
        self.label_psfview.setText(_translate("Phase_retrieval", "PSF view", None))
        self.label_pupilview.setText(_translate("Phase_retrieval", "Retrieved pupil", None))
        self.pushButton_yzview.setText(_translate("Phase_retrieval", "x-z", None))
        self.pushButton_xyview.setText(_translate("Phase_retrieval", "x-y", None))
        self.pushButton_xzview.setText(_translate("Phase_retrieval", "x-z", None))
        self.pushButton_lineview.setText(_translate("Phase_retrieval", "line", None))
        self.checkBox_rm4.setText(_translate("Phase_retrieval", "remove 1-4", None))
        self.pushButton_pfraw.setText(_translate("Phase_retrieval", "Original", None))
        self.pushButton_pffit.setText(_translate("Phase_retrieval", "Fit", None))
        self.pushButton_zbar.setText(_translate("Phase_retrieval", "Z-bar", None))
        self.pushButton_loadpsf.setText(_translate("Phase_retrieval", "Load psf...", None))
        self.label_pupilfname.setText(_translate("Phase_retrieval", "Pupil file name", None))
        self.pushButton_pfbrowse.setText(_translate("Phase_retrieval", "Browse", None))
        self.pushButton_savepupil.setText(_translate("Phase_retrieval", "Save pupil", None))
        self.pushButton_savefit.setText(_translate("Phase_retrieval", "Save z-fit", None))

