# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cat_decorate.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(784, 536)
        self.import_b = QtWidgets.QPushButton(Dialog)
        self.import_b.setGeometry(QtCore.QRect(580, 60, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.import_b.setFont(font)
        self.import_b.setObjectName("import_b")
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(580, 270, 141, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.bow_b = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.bow_b.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.bow_b.setFont(font)
        self.bow_b.setObjectName("bow_b")
        self.verticalLayout.addWidget(self.bow_b)
        self.hat_b = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.hat_b.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.hat_b.setFont(font)
        self.hat_b.setObjectName("hat_b")
        self.verticalLayout.addWidget(self.hat_b)
        self.glasses_b = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.glasses_b.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.glasses_b.setFont(font)
        self.glasses_b.setObjectName("glasses_b")
        self.verticalLayout.addWidget(self.glasses_b)
        self.reset_b = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.reset_b.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.reset_b.setFont(font)
        self.reset_b.setObjectName("reset_b")
        self.verticalLayout.addWidget(self.reset_b)
        self.cat_label = QtWidgets.QLabel(Dialog)
        self.cat_label.setGeometry(QtCore.QRect(50, 40, 450, 450))
        self.cat_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cat_label.setAutoFillBackground(True)
        self.cat_label.setFrameShape(QtWidgets.QFrame.Box)
        self.cat_label.setText("")
        self.cat_label.setObjectName("cat_label")
        self.select_widget = QtWidgets.QWidget(Dialog)
        self.select_widget.setEnabled(True)
        self.select_widget.setGeometry(QtCore.QRect(60, 140, 561, 291))
        self.select_widget.setAutoFillBackground(True)
        self.select_widget.setObjectName("select_widget")
        self.label = QtWidgets.QLabel(self.select_widget)
        self.label.setGeometry(QtCore.QRect(30, 30, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.b1 = QtWidgets.QPushButton(self.select_widget)
        self.b1.setGeometry(QtCore.QRect(30, 90, 100, 100))
        self.b1.setText("")
        self.b1.setObjectName("b1")
        self.b2 = QtWidgets.QPushButton(self.select_widget)
        self.b2.setGeometry(QtCore.QRect(160, 90, 100, 100))
        self.b2.setText("")
        self.b2.setObjectName("b2")
        self.b3 = QtWidgets.QPushButton(self.select_widget)
        self.b3.setGeometry(QtCore.QRect(290, 90, 100, 100))
        self.b3.setText("")
        self.b3.setObjectName("b3")
        self.b4 = QtWidgets.QPushButton(self.select_widget)
        self.b4.setGeometry(QtCore.QRect(420, 90, 100, 100))
        self.b4.setText("")
        self.b4.setObjectName("b4")
        self.b0 = QtWidgets.QPushButton(self.select_widget)
        self.b0.setGeometry(QtCore.QRect(30, 220, 141, 35))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(12)
        self.b0.setFont(font)
        self.b0.setObjectName("b0")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Cat decorate"))
        self.import_b.setText(_translate("Dialog", "????????????..."))
        self.bow_b.setText(_translate("Dialog", "????????????"))
        self.hat_b.setText(_translate("Dialog", "?????????"))
        self.glasses_b.setText(_translate("Dialog", "?????????"))
        self.reset_b.setText(_translate("Dialog", "??????"))
        self.label.setText(_translate("Dialog", "????????????"))
        self.b0.setText(_translate("Dialog", "???????????????"))

