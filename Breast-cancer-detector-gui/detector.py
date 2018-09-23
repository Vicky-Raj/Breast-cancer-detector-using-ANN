import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import tkinter as tk
import tkinter.filedialog as files
from os import getcwd
from os.path import join

data = pd.read_csv(join(getcwd(),'data.csv'))
X = data.iloc[:,2:data.shape[1]+1].values
scaler = StandardScaler()
scaler.fit(X)
model = load_model(join(getcwd(),'detector.h5'))
filepath = ''
savepath = ''
filename = ''

def get_filepath():
    global filepath
    filepath = str(files.askopenfile().name)
    label1['text'] = 'filepath: ' + filepath

def get_savepath():
    global savepath
    savepath = str(files.askdirectory())
    label2['text'] = 'savepath: ' + savepath

def predict_save():
    global filename
    filename = str(entry1.get())
    incomp = pd.read_csv(filepath)
    X_incomp = incomp.iloc[:,1:incomp.shape[1]+1].values
    X_incomp = scaler.transform(X_incomp)
    prediction = model.predict_classes(X_incomp)
    accur = model.predict(X_incomp).max(axis=-1)
    incomp['diagnosis'] = prediction
    incomp['accuracy'] = accur    
    incomp.to_csv(join(savepath,filename + '.csv'))
    label3['text'] = 'saved to: ' + join(savepath,filename + '.csv')
    
    
window = tk.Tk()
window.title('Breast Cancer Detector')
button1 = tk.Button(window,text='filepath',command=get_filepath)
button2 = tk.Button(window,text='savepath',command=get_savepath)
label1 = tk.Label(window,text='filepath'+filepath)
label2 = tk.Label(window,text='savepath'+savepath)
label3 = tk.Label(window)
entry1 = tk.Entry(window)
button3 = tk.Button(window,text='save&predict',command=predict_save)
button1.pack()
label1.pack()
button2.pack()
label2.pack()    
entry1.pack()
button3.pack()
label3.pack()
window.mainloop()




















