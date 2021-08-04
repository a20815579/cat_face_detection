import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from cat_decorate import Ui_Dialog

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from cat_decorate import Ui_Dialog

import os
import torch
import torchvision
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
from math import atan2, degrees

from cat_CNN import CatPicture


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.select_widget.setVisible(False)
        self.show()
        self.ui.import_b.clicked.connect(self.import_b_clicked)
        self.ui.bow_b.clicked.connect(self.bow_b_clicked)
        self.ui.hat_b.clicked.connect(self.hat_b_clicked)
        self.ui.glasses_b.clicked.connect(self.glasses_b_clicked)
        self.ui.reset_b.clicked.connect(self.reset_b_clicked)
        self.ui.b0.clicked.connect(self.b0_clicked)
        self.ui.b1.clicked.connect(self.b1_clicked)
        self.ui.b2.clicked.connect(self.b2_clicked)
        self.ui.b3.clicked.connect(self.b3_clicked)
        self.ui.b4.clicked.connect(self.b4_clicked)
        self.file_name = ""
        self.cat_picture = None
        self.icon_list = [QIcon(QPixmap("C:\\code\\cat_picture\\hat1.png").scaled(
            QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\hat2.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\hat3.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\hat4.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\bow1.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\bow2.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\glasses1.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio)),
            QIcon(QPixmap("C:\\code\\cat_picture\\glasses2.png").scaled(
                QSize(100, 100), Qt.KeepAspectRatio))
        ]
        self.ui.b1.setIconSize(QSize(100, 100))
        self.ui.b2.setIconSize(QSize(100, 100))
        self.ui.b3.setIconSize(QSize(100, 100))
        self.ui.b4.setIconSize(QSize(100, 100))
        self.ui.b3.setIcon(self.icon_list[2])
        self.ui.b4.setIcon(self.icon_list[3])

    def import_b_clicked(self):
        self.file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", os.getcwd(), "Image files (*.jpg *.jpeg *.png)")
        self.cat_picture = CatPicture(self.file_name)
        pixmap = QPixmap(self.file_name).scaled(
            QSize(450, 450), Qt.KeepAspectRatio)
        self.ui.cat_label.setPixmap(pixmap)
        self.ui.bow_b.setEnabled(True)
        self.ui.hat_b.setEnabled(True)
        self.ui.glasses_b.setEnabled(True)
        self.ui.reset_b.setEnabled(True)

    def hat_b_clicked(self):
        self.ui.select_widget.setVisible(True)
        self.ui.b1.setIcon(self.icon_list[0])
        self.ui.b2.setIcon(self.icon_list[1])
        self.ui.b3.setVisible(True)
        self.ui.b4.setVisible(True)
        self.cat_picture.which_d = 0

    def bow_b_clicked(self):
        self.ui.select_widget.setVisible(True)
        self.ui.b1.setIcon(self.icon_list[4])
        self.ui.b2.setIcon(self.icon_list[5])
        self.ui.b3.setVisible(False)
        self.ui.b4.setVisible(False)
        self.cat_picture.which_d = 1

    def glasses_b_clicked(self):
        self.ui.select_widget.setVisible(True)
        self.ui.b1.setIcon(self.icon_list[6])
        self.ui.b2.setIcon(self.icon_list[7])
        self.ui.b3.setVisible(False)
        self.ui.b4.setVisible(False)
        self.cat_picture.which_d = 2

    def reset_b_clicked(self):
        self.cat_picture.wear_list = [0, 0, 0]
        self.CatWearDecorate(0)

    def CatWearDecorate(self, n):
        self.ui.select_widget.setVisible(False)
        self.cat_picture.setWearList(n)
        new_name = self.cat_picture.WearDecorate()
        pixmap = QPixmap(new_name).scaled(
            QSize(450, 450), Qt.KeepAspectRatio)
        self.ui.cat_label.setPixmap(pixmap)

    def b0_clicked(self):
        self.CatWearDecorate(0)

    def b1_clicked(self):
        self.CatWearDecorate(1)

    def b2_clicked(self):
        self.CatWearDecorate(2)

    def b3_clicked(self):
        self.CatWearDecorate(3)

    def b4_clicked(self):
        self.CatWearDecorate(4)


app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
