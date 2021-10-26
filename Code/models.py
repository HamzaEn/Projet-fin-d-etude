from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from keras.preprocessing import image
from keras.models import load_model
from joblib import dump, load

new_model = load_model('arabic_alphabet_98.h5')
def getProbabilitiesCnn(url):
    'cette fonction prend url de image et retourne les probabilitees calculer par le CNN'
    img = Image.open(url).convert('L')
    img=img.resize((32,32))
    img=255-image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img,axis=0)
    p= new_model.predict(img)
    return p

def top_3(p):
    'cette fonction prend comme param les probabilite calculer par CNN et retourne les 3 grande probabilees avec ses index'
    p=np.reshape(p,28)
    MAX=[[1,2,3],[1,2,3]]
    MAX[0][0]=np.max(p)*100
    MAX[1][0]=np.argmax(p)
    p[np.argmax(p)]=-1
    MAX[0][1]=np.max(p)*100
    MAX[1][1]=np.argmax(p)
    p[np.argmax(p)]=-1
    MAX[0][2]=np.max(p)*100
    MAX[1][2]=np.argmax(p)
    return MAX

def labelToText(label):
    'cette fonction prend un nombre entre 0 et 27 et retourne alphabet qui correspond a ce nombre'
    arabic_labels = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', ' د', 'ذ',
                'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 
                'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    return arabic_labels[label]

def first_layer_output(url):
    'cette fonction retourne les feature extrayer par CNN d une image de url donnee en parm'
    from keras import models
    layer_outputs = [layer.output for layer in new_model.layers[:1]]               
    activation_model = models.Model(inputs=new_model.input, outputs=layer_outputs)
    img = Image.open(url).convert('L')
    img=img.resize((32,32))
    img=255-image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img,axis=0)
    activations = activation_model.predict(img)
    return activations
    
    
def predicted_SVM(url):
    img = Image.open(url).convert('L')
    img=img.resize((32,32))
    img=255-image.img_to_array(img)
    img=img.reshape(32,32)
    img, img_hog = hog(img, orientations=4, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True, multichannel=False)
    img = np.expand_dims(img,axis=0)
    clf = load('best_svm.joblib') 
    svm_pred=clf.predict(img)
    svm_pred=svm_pred[0]
    return svm_pred, img_hog
    
    
    
    
    
'''
def main():
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    svm_pred, img_hog=predicted_SVM('C:/Users/MOUAD/Desktop/ahcd1/b.jpg')
    print(svm_pred)
    print(5)
    plt.imshow(img_hog, cmap='gray'), plt.colorbar()
    plt.show()
if __name__ == "__main__":
    main()
    
'''