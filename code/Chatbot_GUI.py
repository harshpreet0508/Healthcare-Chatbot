
# Importing the necessary libraries
import numpy as np                          # mathematical computations
import matplotlib.pyplot as plt             # visualization
import pandas as pd                         # data related operation
from tkinter import *                       # Graphical User Interface
from sklearn.metrics import accuracy_score  # evaluating efficiency

# Importing the necessary algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# List of the symptoms 
symptoms = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

# List of Diseases 
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

# list for matching inputs
l2 = []
for i in range(0,len(symptoms)):
    l2.append(0)

# Loading the training dataset
training_dataset = pd.read_csv("training.csv")

# Replace the diseases with values
training_dataset.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace = True)

# Splitting the training data
X = training_dataset[symptoms]             
y = training_dataset[["prognosis"]]        

# Loading the testing dataset
testing_dataset = pd.read_csv("testing.csv")

# Using inbuilt function replace in pandas for replacing the values
testing_dataset.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)

# Splitting the test data
X_test = testing_dataset[symptoms]
y_test = testing_dataset[["prognosis"]]

data = pd.read_csv("training.csv", index_col = 'prognosis')
def scatterplt(d):
    x = ((data.loc[d]).sum())             # total sum of symptom reported for given disease
    x.drop(x[x==0].index, inplace=True)   # dropping symptoms with values 0
    y = x.keys()                          # storing names of symptoms in y
    plt.title(d)
    plt.scatter(y, x.values)              #  scatter plot  
    plt.show()

def scatterinp(sym1, sym2, sym3, sym4):
    x = [sym1, sym2, sym3, sym4]          # storing input symptoms in y
    y = [0, 0, 0, 0]                      # assigning values to the symptoms
    if(sym1 != 'Select Here'):
        y[0] = 1
    if(sym2 != 'Select Here'):
        y[1] = 1
    if(sym3 != 'Select Here'):
        y[2] = 1
    if(sym4 != 'Select Here'):
        y[3] = 1
    plt.scatter(x,y)                      #  scatter plot  
    plt.show()

# GUI begins 
root = Tk()

# Decision Tree Classifier
pred1 = StringVar()
def DecisionTree():
    
    if len(NameEn.get()) == 0:               # if name is not entered
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Please provide a name")
        if comp:
            root.mainloop()
    
    elif((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")): # if < 2 symptoms 
        pred1.set(" ")
        sym = messagebox.askokcancel("System","Please provide atleast 2 symptoms")
        if sym:
            root.mainloop()
            
    else:                                    # make predictions and plots
        print("Name: ", NameEn.get())

        dt = DecisionTreeClassifier() 
        dt.fit(X,y)

        y_pred = dt.predict(X_test)        
        print("------------------------------")
        print("Algorithm: Decision Tree")
        print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred))) # accuracy on the test set

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]

        for k in range(0,len(symptoms)):
            for z in psymptoms:
                if(z == symptoms[k]):
                    l2[k] = 1

        inputtest = [l2]
        predict = dt.predict(inputtest) # predicting disease acc to symptoms 
        predicted = predict[0]

        h = "no"
        for a in range(0,len(disease)):
            if(predicted == a):
                h = "yes"
                break
    
        if (h == "yes"):
            pred1.set(disease[a])
        else:
            pred1.set("Not Found")
        
        # scatter plot of input symptoms
        scatterinp(Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get())
        
        # scatter plot of disease predicted vs its symptoms
        scatterplt(pred1.get())

# Random Forest Classifier        
pred2 = StringVar()
def randomforest():
    
    if len(NameEn.get()) == 0:       # if name is not entered
        pred1.set(" ")
        comp = messagebox.askokcancel("System","Please provide a name")
        if comp:
            root.mainloop()
            
    elif((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")): # if < 2 symptoms
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Please provide atleast 2 symptoms")
        if sym:
            root.mainloop()
    else:                            # make predictions and plots
        
        rf = RandomForestClassifier(n_estimators = 100)
        rf = rf.fit(X,np.ravel(y))

        # calculating accuracy 
        y_pred = rf.predict(X_test)
        print("------------------------------")
        print("Algorithm: Random Forest")
        print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred))) # accuracy on test set
    
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]

        for k in range(0,len(symptoms)):
            for z in psymptoms:
                if(z == symptoms[k]):
                    l2[k] = 1

        inputtest = [l2]
        predict = rf.predict(inputtest) # predicting disease acc to symptoms 
        predicted = predict[0]

        h = "no"
        for a in range(0, len(disease)):
            if(predicted == a):
                h = "yes"
                break
            
        if (h == "yes"):
            pred2.set(disease[a])
            
        else:
            pred2.set("Not Found")
    
        # printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred2.get())
        
# Naive Bayes        
pred3 = StringVar()
def NaiveBayes():
    
    if len(NameEn.get()) == 0: # if name is not entered
        pred1.set(" ")
        comp = messagebox.askokcancel("System","Please provide a name")
        if comp:
            root.mainloop()
            
    elif((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")): # if < 2 symptoms
        pred1.set(" ")
        sym = messagebox.askokcancel("System","Please provide atleast 2 symptoms")
        if sym:
            root.mainloop()
    else:
        
        nb = GaussianNB()
        nb = nb.fit(X,np.ravel(y))
        
        y_pred = nb.predict(X_test)
        print("------------------------------")
        print("Algorithm: Naive Bayes")
        print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred))) # accuracy on test set
        

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]
        
        for k in range(0,len(symptoms)):
            for z in psymptoms:
                if(z == symptoms[k]):
                    l2[k] = 1

        inputtest = [l2]
        predict = nb.predict(inputtest) # predicting disease acc to symptoms 
        predicted = predict[0]

        h = "no"
        for a in range(0,len(disease)):
            if(predicted == a):
                h = "yes"
                break
            
        if (h == "yes"):
            pred3.set(disease[a])
        else:
            pred3.set("Not Found")
        
        # printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred3.get())
 
# K Neighbours Classifier          
pred4 = StringVar()
def KNN():
    if len(NameEn.get()) == 0:         # if name is not entered
        pred1.set(" ")
        comp = messagebox.askokcancel("System","Please provide a name")
        if comp:
            root.mainloop()
            
    elif((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")): # if < 2 symptoms
        pred1.set(" ")
        sym = messagebox.askokcancel("System","Please provide atleast 2 symptoms")
        if sym:
            root.mainloop()
            
    else:                               # make predictions and plots
        knn = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski" ,p = 2)
        knn = knn.fit(X,np.ravel(y))
    
        
        y_pred = knn.predict(X_test)
        print("------------------------------")
        print("Algorithm: K Nearest Neighbor")
        print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred))) # accuracy on test set

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]

        for k in range(0,len(symptoms)):
            for z in psymptoms:
                if(z == symptoms[k]):
                    l2[k] = 1

        inputtest = [l2]
        predict = knn.predict(inputtest) # predicting disease acc to symptoms 
        predicted = predict[0]

        h = 'no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h = "yes"
                break
            
        if (h == "yes"):
            pred4.set(disease[a])
        else:
            pred4.set("Not Found")
        
        # printing scatter plot of disease predicted vs its symptoms
        scatterplt(pred4.get())
        print("----------------------------------")

# Tk class is used to create a root window
root.configure(background = "Ivory")
root.title("DISEASE PREDICTOR")
root.resizable(0,0)

# taking symptoms as input
Symptom1 = StringVar()
Symptom1.set("Select Here")

Symptom2 = StringVar()
Symptom2.set("Select Here")

Symptom3 = StringVar()
Symptom3.set("Select Here")

Symptom4 = StringVar()
Symptom4.set("Select Here")

# Headings for the GUI 
L1 = Label(root, text = "\tHEATHCARE CHATBOT", fg = "Red", bg = "Ivory")
L1.config(font = ("Times", 20, "bold italic"))
L1.grid(row = 1, column = 0, columnspan = 2, padx = 100, pady = 15)

# Label for the name
Name = Label(root, text = "Patient's Name", fg = "Red", bg = "Ivory")
Name.config(font=("Times", 13, "bold italic"))
Name.grid(row = 6, column = 0, padx = 5, pady = 15)

# Labels for the symptoms
S1lb = Label(root, text = "Symptom 1", fg = "Black", bg = "Ivory")
S1lb.config(font=("Times", 13, "bold italic"))
S1lb.grid(row = 7, column = 0, padx = 10, pady = 10)

S2lb = Label(root, text = "Symptom 2", fg = "Black", bg = "Ivory")
S2lb.config(font=("Times", 13, "bold italic"))
S2lb.grid(row = 8, column = 0, padx = 10, pady = 10)

S3lb = Label(root, text = "Symptom 3", fg = "Black", bg = "Ivory")
S3lb.config(font=("Times", 13, "bold italic"))
S3lb.grid(row = 9, column = 0, padx = 10, pady = 10)

S4lb = Label(root, text = "Symptom 4", fg = "Black", bg = "Ivory")
S4lb.config(font=("Times", 13, "bold italic"))
S4lb.grid(row = 10, column = 0, padx = 10, pady = 10)

# Labels for the algorithms
L1dt = Label(root, text = "Decision Tree", bg = "sky blue", fg = "blue", width = 17)
L1dt.config(font=("Times", 13, "bold italic"))
L1dt.grid(row = 15, column = 0, padx = 10, pady = 10)

L2rf = Label(root, text = "Random Forest", bg = "yellow", fg = "red", width = 17)
L2rf.config(font=("Times", 13, "bold italic"))
L2rf.grid(row = 17, column = 0, padx = 10, pady = 10)

L3nb = Label(root, text = "Naive Bayes", bg = "Light green", fg = "red", width = 17)
L3nb.config(font=("Times", 13, "bold italic"))
L3nb.grid(row = 19, column = 0, padx = 10, pady = 10)

L4knn = Label(root, text = "K Nearest Neighbour", bg = "light salmon", fg = "purple", width = 17)
L4knn.config(font=("Times", 13, "bold italic"))
L4knn.grid(row = 21, column = 0, padx = 10, pady = 10)

# Entry for name
NameEn = Entry(root, textvariable = Name, bg = "ivory2", width = 25)
NameEn.grid(row = 6, column = 1)

# drop down menu for symptoms
op = sorted(symptoms)

S1 = OptionMenu(root, Symptom1, *op)
S1.grid(row = 7, column = 1)

S2 = OptionMenu(root, Symptom2, *op)
S2.grid(row = 8, column = 1)

S3 = OptionMenu(root, Symptom3, *op)
S3.grid(row = 9, column = 1)

S4 = OptionMenu(root, Symptom4, *op)
S4.grid(row = 10, column = 1)

# Buttons for predicting the disease using the algorithms
dst = Button(root, text = "Prediction 1", command = DecisionTree, bg = "light salmon", fg = "purple")
dst.config(font=("Times", 13, "bold italic"))
dst.grid(row = 7, column = 2, padx = 20, pady = 15)

rnf = Button(root, text = "Prediction 2", command = randomforest, bg = "Light green", fg = "red")
rnf.config(font=("Times", 13, "bold italic"))
rnf.grid(row = 8, column = 2, padx = 20, pady = 15)

lr = Button(root, text = "Prediction 3", command = NaiveBayes, bg = "yellow", fg = "red")
lr.config(font=("Times", 13, "bold italic"))
lr.grid(row = 9, column = 2, padx = 20, pady = 15)

kn = Button(root, text = "Prediction 4", command=KNN, bg="sky blue", fg="blue")
kn.config(font=("Times", 13, "bold italic"))
kn.grid(row = 10, column = 2, padx = 20, pady = 15)

# Predicting from different algorithms
p1 = Label(root, font = ("Times", 13, "bold italic"),text = "Decision Tree",height = 1, bg = "Light salmon"
         ,width = 30,fg = "purple",textvariable = pred1, relief = "sunken").grid(row = 15, column = 1, padx = 5, pady = 10)

p2 = Label(root, font = ("Times", 13, "bold italic"),text = "Random Forest",height = 1,bg = "light green"
         ,width = 30,fg = "red",textvariable = pred2, relief = "sunken").grid(row = 17, column = 1, padx = 5, pady = 10)

p3 = Label(root, font = ("Times", 13, "bold italic"), text = "Naive Bayes", height = 1, bg = "yellow"
         ,width = 30,fg = "red", textvariable = pred3, relief = "sunken").grid(row = 19, column = 1, padx = 5, pady = 10)

p4 = Label(root, font = ("Times", 13, "bold italic"), text = "K Nearest Neighbour", height = 1, bg = "sky blue"
         ,width = 30,fg = "blue", textvariable = pred4, relief = "sunken").grid(row = 21, column = 1, padx = 5, pady = (10,15))

# to run the Tkinter event loop
root.mainloop()



