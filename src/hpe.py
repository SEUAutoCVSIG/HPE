# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hpe.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

'''
    Created on wed Sept 6 19:34 2018

    Author           : Heng Tan
    Email            : 1608857488@qq.com
    Last edit date   : Sept 27 1:23 2018

South East University Automation College, 211189 Nanjing China

The following codes referenced Ayoosh Kathuria's blog:
UI design
'''

from PyQt5 import QtCore, QtGui, QtWidgets
from  PyQt5.QtWidgets import QWidget,QPushButton,QMessageBox,QLineEdit,QApplication
import sys, os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

# UI函数直接放在Ui_HPE类里面 下面有标识

class Ui_HPE(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self, HPE):
        HPE.setObjectName("HPE")
        HPE.resize(942, 653)
        self.centralWidget = QtWidgets.QWidget(HPE)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 180, 191, 41))
        self.pushButton.setStyleSheet("")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(80, 240, 191, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_3.setGeometry(QtCore.QRect(80, 310, 191, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_4.setGeometry(QtCore.QRect(80, 380, 191, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_5.setGeometry(QtCore.QRect(80, 440, 191, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_6.setGeometry(QtCore.QRect(600, 240, 261, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_7.setGeometry(QtCore.QRect(600, 180, 261, 41))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_8.setGeometry(QtCore.QRect(600, 300, 261, 41))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_9.setGeometry(QtCore.QRect(640, 440, 191, 41))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(70, 0, 831, 171))
        self.label.setStyleSheet("font: 14pt \"Arial\";")
        self.label.setObjectName("label")
        HPE.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(HPE)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 942, 26))
        self.menuBar.setObjectName("menuBar")
        HPE.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(HPE)
        self.mainToolBar.setObjectName("mainToolBar")
        HPE.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(HPE)
        self.statusBar.setObjectName("statusBar")
        HPE.setStatusBar(self.statusBar)

        self.retranslateUi(HPE)
        QtCore.QMetaObject.connectSlotsByName(HPE)

        self.pushButton.clicked.connect(self.HumanDetector)
        self.pushButton_2.clicked.connect(self.SingleEstimation)
        self.pushButton_3.clicked.connect(self.MutiPersonEstimation)
        self.pushButton_4.clicked.connect(self.CocoDisplay)
        self.pushButton_5.clicked.connect(self.MpiiDisplay)
        self.pushButton_6.clicked.connect(self.DemoHD)
        self.pushButton_7.clicked.connect(self.DemoSE)
        self.pushButton_8.clicked.connect(self.DemoMSE)
        self.pushButton_9.clicked.connect(self.TrickMode)

        HPE.show()

    def retranslateUi(self, HPE):
        _translate = QtCore.QCoreApplication.translate
        HPE.setWindowTitle(_translate("HPE", "HPE"))
        self.pushButton.setText(_translate("HPE", "Human Detector"))
        self.pushButton_2.setText(_translate("HPE", "Single Estimation"))
        self.pushButton_3.setText(_translate("HPE", "Multi-person Estimation"))
        self.pushButton_4.setText(_translate("HPE", "COCO Display"))
        self.pushButton_5.setText(_translate("HPE", "MPII Dispay"))
        self.pushButton_6.setText(_translate("HPE", "Demo - Single Estimation"))
        self.pushButton_7.setText(_translate("HPE", "Demo - Human Detector"))
        self.pushButton_8.setText(_translate("HPE", "Demo - Muti-person Estimation"))
        self.pushButton_9.setText(_translate("HPE", "Trick Mode"))
        self.label.setText(_translate("HPE", "Human Poses Estimation on Convolutional Neural Network and Deep Learning\n"
""))

# 以下为HPE函数编写 函数内容需更改

    def HumanDetector(self):
        reply = QMessageBox.information(self,'FUCK','BULLSHIT',
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.la.setText('你选择了Yes！')
        elif reply == QMessageBox.No:
            self.la.setText('你选择了No！')

    def SingleEstimation(self):
        self.statusBar.showMessage('Fuck')
    def MutiPersonEstimation(self):
        self.statusBar.showMessage('Fuck')
    def CocoDisplay(self):
        self.statusBar.showMessage('Fuck')
    def MpiiDisplay(self):
        self.statusBar.showMessage('Fuck')
    def DemoHD(self):
        self.statusBar.showMessage('Fuck')
    def DemoSE(self):
        self.statusBar.showMessage('Fuck')
    def DemoMSE(self):
        self.statusBar.showMessage('Fuck')
    def TrickMode(self):
        self.statusBar.showMessage('Fuck')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = QMainWindow()
    w = Ui_HPE()
    w.setupUi(form)
    form.show()
    sys.exit(app.exec_())

