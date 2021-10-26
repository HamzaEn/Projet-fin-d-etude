# import Important modules
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType 
from pyqtgraph import PlotWidget, plot
import PyQt5
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os
import sys
import time
import math
import models 
import copy
import sys


# import UI file
FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__),'interface.ui'))


class mainApp(QMainWindow, FORM_CLASS):
    
    
    SVM_HOG= []
    SVM_pred=0
    url_dir= []
    i=0
    pred=[]
    def __init__(self, parent=None):
        super(mainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setFixedSize(756, 682)
        self.setupUi(self)
        self.setWindowTitle("Arabic Alphabets Classifier")
        self.setWindowIcon(QIcon('icon.png'))
        self.handle_button()
        self.progressBar.setValue(0)
        self.progressBar_2.setValue(0)
        self.progressBar_3.setValue(0)
        self.progressBar_4.setValue(0)
        self.handle_button2()
        self.handle_button3()
        self.handle_button4() 
    
    def handle_button(self):
        self.pushButton.clicked.connect(self.select_image)
        
    def select_image(self):
        self.url_dir = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.jpg *.png *.jpeg)')
        if(self.url_dir[0]):
            self.i=1
            self.label.setPixmap(QPixmap(self.url_dir[0]).scaled(131, 131))
            self.pred=models.getProbabilitiesCnn(self.url_dir[0])
            p=models.top_3(copy.deepcopy(self.pred))
            self.pred=np.reshape(self.pred,28)
            self.SVM_pred, self.SVM_HOG=models.predicted_SVM(self.url_dir[0])
            self.label_4.setText(models.labelToText(p[1][0]))
            self.label_5.setText(models.labelToText(p[1][1]))
            self.label_6.setText(models.labelToText(p[1][2]))
            self.label_15.setText(models.labelToText(self.SVM_pred))
            for i in range(100):
                if(i<=p[0][0]):
                    self.progressBar.setValue(i)
                if(i<=p[0][1]):
                    self.progressBar_2.setValue(i)
                if(i<=p[0][2]):
                    self.progressBar_3.setValue(i)
                if(i<=100):
                    self.progressBar_4.setValue(i)
                time.sleep(0.01)
            
    def handle_button2(self):
        self.pushButton_3.clicked.connect(self.aff_image)
        self.lineEdit.returnPressed.connect(self.aff_image)
    def handle_button3(self):
        self.pushButton_2.clicked.connect(self.aff_hist)
        
    def aff_hist(self):
        if self.i != 0 :
            alphabet=['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', ' د', 'ذ',
                    'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 
                    'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
            plt.bar(alphabet,self.pred)
            plt.show()
    def aff_image(self):
        if self.i != 0 :
            nb=self.lineEdit.text()
            try:
                nb=int(nb)
            except ValueError:
                self.lineEdit.setText('"0<=nb<32"')
                return
            if(nb<0 or nb >31):
                self.lineEdit.setText('"You should put number between 0 and 31"')
                return 
            m=models.first_layer_output(self.url_dir[0])
            plt.imshow(m[0, :, :, nb], cmap='viridis')
            plt.show()

    
    def handle_button4(self):
        self.pushButton_4.clicked.connect(self.aff_hog)
        
    def aff_hog(self):
        if self.i != 0 :
            plt.imshow(self.SVM_HOG, cmap='gray')
            plt.show()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Arabic Alphabets Classifier")
    window = mainApp()
    
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()