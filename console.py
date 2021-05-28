## A Healthcare Chatbot to simulate the predictions of a General Physician ##

# Importing the necessary libraries
import numpy as np                   # mathematical computations
import matplotlib.pyplot as plt      # visualization
import pandas as pd                  # data related operation

# Loading the dataset
training_dataset = pd.read_csv('Training.csv')

# Splitting the data into feature matrix and vector of predictions
X = training_dataset.iloc[:,0:132].values
y = training_dataset.iloc[:,-1].values

# Fetching unique diseases
dim_red = training_dataset.groupby(training_dataset['prognosis']).max()

# Encoding the strings in y to integer
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)

# Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

# Applying the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

# Saving the columns
cols = training_dataset.columns
cols = cols[:-1]
features = cols

# Importing a visual tree
from sklearn.tree import _tree

# function to simulate the working of chatbot
def execute_bot():
    
    print('Please reply with Yes or No for following symptoms ')
    
    def get_disease(node):
        
        node = node[0]
        val = node.nonzero()
        disease = lab.inverse_transform(val[0]);
        return disease

    def tree_to_code(tree,feature_names):
        
        tree_ = tree.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED 
                            else 'undefined' for i in tree_.feature]
        
        sym_present = []
        def recurse(node,depth):
               
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node];
                threshold = tree_.threshold[node]
                
                print(name + '?', end =' ') # asking for symptoms
                
                a = input();
                a = a.lower();
                
                if a == 'yes':
                    val = 1
                
                else:
                    val = 0
                    
                if val <= threshold:
                    recurse(tree_.children_left[node],depth+1)
                
                else:
                    sym_present.append(name)
                    recurse(tree_.children_right[node],depth+1)
         
            else:
                disease = get_disease(tree_.value[node]);
                
                print('You may have ' + disease) # predicting disease
                print()
                
                red = dim_red.columns
                sym_given = red[dim_red.loc[disease].values[0].nonzero()]
                
                print('Symptoms present '+ str(list(sym_present)))
                print()
                
                print('Symptoms given '+ str(list(sym_given)))
                print()
                
                conf = (1.0 * len(sym_present))/len(sym_given)
                print('Confidence Level is ' + str(round(conf,3)))
                print()
                
                print('The model suggests: ') # suggesting doctors
                
                r = doctors[doctors['disease'] == disease[0]]
                print('*Consult ' + str(r['name'].values)) 
                
                print('*Visit ' + str(r['link'].values))
                
        recurse(0,1)
        
    tree_to_code(dt,cols)

# Importing to doctors dataset
doctors_dataset = pd.read_csv('doctors_dataset.csv',names = ['Name','Description'])

# Fetching the unique diseases
diseases = dim_red.index
diseases = pd.DataFrame(diseases)

# Dataset linking doctor and disease
doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']
doctors['name'] = doctors_dataset['Name']
doctors['link'] = doctors_dataset['Description']

# Execute the chatbot
execute_bot()

