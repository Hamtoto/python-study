# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BLNC(object):
    def setupUi(self, BLNC):
        BLNC.setObjectName("BLNC")
        BLNC.resize(800, 600)
        BLNC.setMinimumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(BLNC)
        self.centralwidget.setObjectName("centralwidget")
        BLNC.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(BLNC)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        BLNC.setMenuBar(self.menubar)
        self.actionSet_API_KEY = QtWidgets.QAction(BLNC)
        self.actionSet_API_KEY.setObjectName("actionSet_API_KEY")
        self.actionHelp = QtWidgets.QAction(BLNC)
        self.actionHelp.setObjectName("actionHelp")
        self.menu.addAction(self.actionSet_API_KEY)
        self.menu.addSeparator()
        self.menu.addAction(self.actionHelp)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(BLNC)
        QtCore.QMetaObject.connectSlotsByName(BLNC)

    def retranslateUi(self, BLNC):
        _translate = QtCore.QCoreApplication.translate
        BLNC.setWindowTitle(_translate("BLNC", "MainWindow"))
        self.menu.setTitle(_translate("BLNC", "설정"))
        self.actionSet_API_KEY.setText(_translate("BLNC", "Set API KEY"))
        self.actionHelp.setText(_translate("BLNC", "Help"))
