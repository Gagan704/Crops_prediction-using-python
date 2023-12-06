from __future__ import print_function

import warnings

import pandas as pd
from sklearn import metrics
##import seaborn as sns
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
##N=0
##P=0
##K=0

df = pd.read_csv("crop_recommendation.csv")
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']

# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


##random forest algorith 
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


### Cross validation score (Random Forest)
##score = cross_val_score(RF,features,target,cv=5)
##
##print(score)


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


###########################################################################################################################


from tkinter import *
from tkinter import ttk


#




root = Tk()
root.title('Crop Recommendation window')
root.geometry('850x650')
root.configure(background="purple2")
var = StringVar()
label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="purple2")
var.set('Crop Recommendation System')
label.grid(row=0,columnspan=6)

label_1 = ttk.Label(root, text ='nitrogen',font=("Helvetica", 16),background="Purple3")
label_1.grid(row=11,column=0)

##label_n = ttk.Label(root, text =csv,font=("Helvetica", 16),background="Purple3")
##label_n.grid(row=11,column=1)
    
Entry_1= Entry(root)
Entry_1.grid(row=11,column=1)

label_2 = ttk.Label(root, text ='phosporus',font=("Helvetica", 16),background="Purple3")
label_2.grid(row=12,column=0)

##label_p = ttk.Label(root, text =csva,font=("Helvetica", 16),background="Purple3")
##label_p.grid(row=12,column=1)
    
Entry_2 = Entry(root)
Entry_2.grid(row=12,column=1)
##    
    
label_3 = ttk.Label(root, text ='pottasium',font=("Helvetica", 16),background="Purple3")
label_3.grid(row=13,column=0)

##label_k = ttk.Label(root, text =csvb,font=("Helvetica", 16),background="Purple3")
##label_k.grid(row=13,column=1)
    
Entry_3 = Entry(root)
Entry_3.grid(row=13,column=1)

label_4 = ttk.Label(root, text ='temperature',font=("Helvetica", 16),background="Purple3")
label_4.grid(row=14,column=0)

##label_1 = ttk.Label(root, text =csvc,font=("Helvetica", 16),background="Purple3")
##label_1.grid(row=14,column=1)
    
Entry_4= Entry(root)
Entry_4.grid(row=14,column=1)


label_5 = ttk.Label(root, text ='humidity',font=("Helvetica", 16),background="Purple3")
label_5.grid(row=15,column=0)

##label_1 = ttk.Label(root, text =csvd,font=("Helvetica", 16),background="Purple3")
##label_1.grid(row=15,column=1)
    
Entry_5 = Entry(root)
Entry_5.grid(row=15,column=1)


label_6 = ttk.Label(root, text ='ph',font=("Helvetica", 16),background="Purple3")
label_6.grid(row=16,column=0)
    
Entry_6 = Entry(root)
Entry_6.grid(row=16,column=1)


label_7 = ttk.Label(root, text ='rainfall',font=("Helvetica", 16),background="Purple3")
label_7.grid(row=17,column=0)
    
Entry_7 = Entry(root)
Entry_7.grid(row=17,column=1)



def predict():
    N = Entry_1.get()
    P = Entry_2.get()
    K = Entry_3.get()
    temperature =Entry_4.get()
    humidity =Entry_5.get()
    ph =Entry_6.get()  
    rainfall = Entry_7.get()   
    out = RF.predict([[float(N),
       float(P),
       float(K),
       float(temperature),
       float(humidity),
       float(ph),
       float(rainfall)]])     ##float(area)
    
    output.delete(0,END)
    output.insert(0,out[0])
   

b1 = Button(root, text = 'predict',font=("Helvetica", 16),background="Purple3",command = predict)
b1.grid(row=20,column=0)
    

output = Entry(root)
output.grid(row=20,column=1)
    
root.mainloop()






#############################################################################################################






