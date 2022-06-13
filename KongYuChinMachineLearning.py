#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression

np.random.seed(1)


# In[2]:


# load data. Put the given datasets on the same directory as this python file. 
X_train=np.array(pd.read_csv('X_train.csv', header=None))
y_train=np.array(pd.read_csv('y_train.csv', header=None))
X_test=np.array(pd.read_csv('X_test.csv', header=None))


#  # Data exploration

# In[3]:


#Finding the size of datasets
print('Data Size of each datasets')
print('X_train has {} rows and {} columns'.format(*X_train.shape)) #Print X_train shape
print('y_train has {} rows and {} columns'.format(*y_train.shape)) #Print y_train shape


# In[4]:


#Find 5 rows from the X data set
pd.DataFrame(X_train).head()


# In[5]:


#Find 5 rows from the y data set
pd.DataFrame(y_train).head()


# In[6]:


#Finding the data type
print('Data Type')
print(type(X_train))
print()
pd.DataFrame(X_train).info() #Finding the data type of X
print()
pd.DataFrame(y_train).info() #Finding the data type of y


# In[7]:


#Finding X data set statistics
pd.DataFrame(X_train).describe()


# In[8]:


#Find The maximum of std
np.max(pd.DataFrame(X_train).describe()[2:3].values)


# In[9]:


#Finding y data set statistics
pd.DataFrame(y_train).describe()


# In[10]:


#Finding the variance of the dataset
print('Variance')
var = {'X_train_variance':np.var(X_train), 'y_train_variance':np.var(y_train)} #Use Library to store variances of X train and y train
for x, vari in var.items():
    print(f"{x}: {vari}")
    
#Compare to X and y 
print('The Largest Variance is: ', max(var.keys()), max(var.values())) #Print the maximum variance 
print('The Smallest Variance is: ', min(var.keys()), min(var.values())) #Print the maximum variance 
print()

#Finding the standard deviation of the dataset
print('standard deviation')
std = {'X_train_standard deviation': np.std(X_train),'y_train_variance':np.std(y_train)}#Use Library to store Std of X train and y train
for y, stds in std.items():
    print(f"{y}: {stds}")
    
#Compare to X and y 
print('The Largest Standard Deviation is: ', max(std.keys()), max(std.values())) #Print the maximum Std 
print('The Smallest Standard Deviation is: ', min(std.keys()), min(std.values())) #Print the maximum Std
print()


# In[11]:


# Finding the missing value of the dataset
print('Missing Value')
if all(pd.DataFrame(X_train).isnull()) == True: #Check X train data set has missing value
    print('X_train dataset has Missing Value')
    print(pd.DataFrame(X_train).isnull().sum(), 'count missing value :', np.sum(pd.DataFrame(X_train).isnull().sum()))
else: #Check X train data set has not missing value
    print('X_train dataset has No Missing Value')
    print(pd.DataFrame(X_train).isnull().sum(),'\ncount missing value :', np.sum(pd.DataFrame(X_train).isnull().sum()))

if all(pd.DataFrame(y_train).isnull()) == True: #Check y train data set has missing value
    print('y_train dataset has Missing Value') 
    print(pd.DataFrame(y_train).isnull().sum(), 'count missing value :',np.sum(pd.DataFrame(y_train).isnull().sum()))
else: #Check y train data set has not missing value
    print('y_train dataset has No Missing Value') 
    print(pd.DataFrame(X_train).isnull().sum(),'\ncount missing value :',np.sum(pd.DataFrame(y_train).isnull().sum()))


# In[12]:


#Matrix of Correlation
plt.title("Correlation Matrix of X data set")
plt.matshow(pd.DataFrame(X_train).corr(), fignum=0) #plot matrix of correlation graph
plt.colorbar() #Check the corrleation

# Create new matric of Corrlation to check whether two matrix of correltion are the same
plt.figure()
cov=np.cov(X_train.T)
stds=np.std(X_train, axis=0)
stds_matrix=np.array([[stds[i]*stds[j] for j in range(20)]for i in range(20)])
new_corr = cov/stds_matrix

plt.title("New Correlation Matrix of X data set")
plt.matshow(new_corr, fignum=0)
plt.colorbar()


# In[13]:


#Check X train dataset whether is normal and have imbalance data
def normal_dis(X):
    plt.figure()
    fig, ax = plt.subplots()
    ax.hist(X, bins=35, density=True, rwidth=1,facecolor='c',alpha=0.5)
    x = np.linspace(-10, 10, 4000)
    ax.plot(x, 1 / np.sqrt(2*np.pi) * np.exp(-(x**2)/2), linewidth=1.8)
    plt.show()


# In[14]:


normal_dis(X_train)


# In[15]:


def train(X, y):
	return np.linalg.inv(X.T@X)@X.T@y # returns the fitted parameter, by the closed-form formula 

params=train(X_train,y_train)
print(params)


# In[16]:


# we first implement a function that makes prediction. I.e., given a dataset (each row of which is an example; with the extra column of all ones) and the fitted parameters, predict the y values of each example
def predict(X_train, params):
    return X_train@params

# find SSE
def SSE(observe, predict):
    return np.sum((observe-predict)**2)
pred=predict(X_train,params)
print(SSE(y_train,pred))

# plot the data and the regression line. Check the relationship between X train and y train (whether is linear)
import matplotlib.pyplot as plt
for n in np.arange(20):
    plt.figure()
    plt.scatter(X_train[:,n],y_train)
    plt.plot(X_train[:,n],pred)


#  # Data preprocessing

# In[17]:


# TODO: Your code for data preprocessing should be inserted here. Create more cells as you see fit. 
# NOTE: You may use data preprocessing methods that are not covered in this course.
# Suggestions: standardization, normalization, k-means...
# Exercise your creativity in this part! There is no one-size-fits-all approach to do data preprocessing for all kinds of datasets. You'll need to experiment with many possibilities. For example, you may even remove some features if you think doing so would increase your models' prediction accuracy.
from sklearn.model_selection import train_test_split
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# split the dataset randomly. That is, I randomly select 30% of our entire dataset for testing purposes and the remaining 70% for training.
X_train_training, X_test_testing, y_train_training, y_test =  train_test_split(X_train, y_train, test_size=0.3, random_state = 0)
print(f'X_train_training = {X_train_training.shape}')
print(f'X_test_testing = {X_test_testing.shape}')
print(f'y_train_training = {y_train_training.shape}')
print(f'y_test  = {y_test.shape}')


# ### Data Preporcessing-standardize

# In[18]:


#Standardize X_train_training and testing data sets.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_training)
X_test_sc = sc.transform(X_test_testing)
pd.DataFrame(X_train_sc).head()
#Check whether normal and have imbalance data
normal_dis(X_train_sc)
pd.DataFrame(X_train_sc).head()


# ### Data Preprocessing-Normalize

# In[19]:


#Normalize X_train_training and testing data sets.
from sklearn import preprocessing
X_train_normali = preprocessing.normalize(X_train_training)
X_test_normali = preprocessing.normalize(X_test_testing)
#Check whether normal and have imbalance data
normal_dis(X_train_normali)
pd.DataFrame(X_train_normali).head()


# ### Data Preprocessing K-Mean

# In[20]:


from sklearn.cluster import KMeans


# In[21]:


#Find the best n
from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X_train)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs, marker='o', color='b')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[22]:


#Test K-Means accuracy and ineritria
from sklearn.preprocessing import LabelEncoder
kmeans=KMeans(n_clusters=6, random_state=0) 
kmeans.fit(X_train)
X_transform=kmeans.transform(X_train)

print(X_transform.shape)
le = LabelEncoder()
le.fit(y_train)
y = le.transform(y_train)

labels = kmeans.labels_
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[23]:


kmeans.inertia_


# ###### In below Model building, I will not use K-Means Clustering to build and analyse because it has low accuracy in the data set, high inertia and big dimensions.

# # Model building & model evaluation

# In[24]:


# TODO: Your code for model building & model evaluation should be inserted here. Create more cells as you see fit. 
#Import Sklearn Model
def Accuracy(search, X):
    print(f'The best parameters found are: {search.best_params_}') # the parameter settings that produce the highest mean cross-validated accuracy
    print(f'The mean cross-validated accuracy is: {search.best_score_}') # shows the mean cross-validated accuracy for the model with the best parameter setting
    print(f'predictions on the few examples under the best parameter setting: {search.predict(X)[:40]}') #use the predict() method of the tree_grid_search object to find the prediction of each example
    print(f'accuracy of the best model on the whole dataset: {accuracy_score(y_train_training, search.predict(X))}')


# In[25]:


class Model:
    def __init__(self, name):
        self.name = name #Set Classifier
        
    def Dataset(self,test, train):
        self.pred_test = self.name.predict(test) 
        print(f"Predicted class labels of the data in test set: \n{self.pred_test}")
        self.pred_train = self.name.predict(train)  #make prediction on the test data set
        print(f"Predicted class labels of the data in train set: \n{self.pred_train}")
    
    def Pred_accur(self):
        print(f'accuracy on train set: {accuracy_score(y_train_training, self.pred_train)}')
        print(f'accuracy on test set: {accuracy_score(y_test, self.pred_test)}')
    
    def ConfusionMatrics(self):
        #plot the Confusion of Matric and report to see precision, recall and f score
        confmat = confusion_matrix(y_test, self.pred_test)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
        plt.xlabel('predicted label')        
        plt.ylabel('true label')
        plt.show()
    
        print(classification_report(y_test, self.pred_test))


# # Decision tree

# In[26]:


tree_param_grid = [ {'min_samples_split': range(2, 20), 'max_depth': range(2, 20)} ]# create a list that stores the parameter settings we want to explore with.
tree_grid_search = GridSearchCV( DecisionTreeClassifier(), tree_param_grid, cv=5) # this code sets up the parameter setting.

X = X_train_training #Let X
y = y_train_training #Let y


# In[27]:


tree_grid_search.fit(X, y) # this code will do the heavy lifting


# In[28]:


Accuracy(tree_grid_search, X) #Find the accuracy of grid


# In[29]:


#Check out the official doc for the complete list of parameters of the model. 
#change to "entropy" by changing the 'criterion' parameter. 
tree = DecisionTreeClassifier(criterion='entropy', max_depth = 9, min_samples_split=12, random_state=0)
tree.fit(X_train_training, y_train_training) #build the decision tree by giving the train datasets (both X_train and y_train) as arguments.


# In[30]:


a=Model(tree) #Call the class (Model) and rename as a
a.Dataset(X_test_testing, X_train_training) #Call the method a.Dataset
a.Pred_accur()


# In[31]:


a.ConfusionMatrics() #Call the method a.ConfusionMatrics


# ### Decision Tree (Standardize)

# In[32]:


tree_param_grid = [ {'min_samples_split': range(2, 20), 'max_depth': range(2, 20)} ]# create a list that stores the parameter settings we want to explore with.
tree_grid_search_sc = GridSearchCV( DecisionTreeClassifier(), tree_param_grid, cv=5) # this code sets up the parameter setting.

X_sc = X_train_sc


# In[33]:


tree_grid_search_sc.fit(X_sc, y)# this code will do the heavy lifting


# In[34]:


Accuracy(tree_grid_search_sc, X_sc)#Find the accuracy of grid


# In[35]:


tree = DecisionTreeClassifier(criterion='entropy', max_depth = 8, min_samples_split=11, random_state=0)
tree.fit(X_train_training, y_train_training)


# In[36]:


a=Model(tree)#Call the class (Model) and rename as a
a.Dataset(X_test_sc, X_train_sc)#Call the method a.Dataset
a.Pred_accur()#Call the method a.Pred_accur


# In[37]:


a.ConfusionMatrics()#Call the method a.ConfusionMatrics


# ### Decision Tree (Normalize)

# In[38]:


tree_param_grid = [ {'min_samples_split': range(2, 20), 'max_depth': range(2, 20)} ]

tree_grid_search_normali = GridSearchCV( DecisionTreeClassifier(), tree_param_grid, cv=5) 

X_normali = X_train_normali


# In[39]:


tree_grid_search_normali.fit(X_normali, y)# this code will do the heavy lifting


# In[40]:


Accuracy(tree_grid_search_normali, X_normali)#Find the accuracy of grid


# In[41]:


tree = DecisionTreeClassifier(criterion='entropy', max_depth = 8, min_samples_split=7, random_state=0)
tree.fit(X_train_normali, y_train_training)


# In[42]:


a=Model(tree)#Call the class (Model) and rename as a
a.Dataset(X_test_normali, X_train_normali)#Call the method a.Dataset
a.Pred_accur()#Call the method a.Pred_accur


# In[43]:


a.ConfusionMatrics()#Call the method a.ConfusionMatrics


# # Naive Bayes

# In[44]:


NB = GaussianNB() #instantiate a model NB
NB.fit(X_train_training, y_train_training)


# In[45]:


a=Model(NB)
a.Dataset(X_test_testing, X_train_training)
a.Pred_accur()


# In[46]:


a.ConfusionMatrics()


# In[47]:


np.mean(cross_val_score(NB, X_train_training, y_train_training, cv=10)) #find the mean accuracy for NB


# ### Navie Bayes (Standarize)

# In[48]:


a=Model(NB)
a.Dataset(X_test_sc, X_train_sc)
a.Pred_accur()


# In[49]:


a.ConfusionMatrics()


# In[50]:


np.mean(cross_val_score(NB, X_train_sc, y_train_training, cv=10))


# ### Navie Bayes (Normalize)

# In[51]:


a=Model(NB)
a.Dataset(X_test_normali, X_train_normali)
a.Pred_accur()


# In[52]:


a.ConfusionMatrics()


# In[53]:


np.mean(cross_val_score(NB, X_train_normali, y_train_training, cv=10))


# # kNN

# In[54]:


kNN_param_grid = [  {'n_neighbors': range(1, 70)} ] # find the best k
kNN_grid_search = GridSearchCV(KNeighborsClassifier(), kNN_param_grid, cv=20)


# In[55]:


kNN_grid_search.fit(X_train_training, y_train_training)


# In[56]:


print(f'The best parameters found are: {kNN_grid_search.best_params_}')
print(f'The mean cross-validated accuracy is: {kNN_grid_search.best_score_}')

kNN_grid_search.predict(X)[:50]
print(f'accuracy of the best model on the whole dataset: {accuracy_score(y_train_training, kNN_grid_search.predict(X))}')


# In[57]:


kNN = KNeighborsClassifier(n_neighbors=21, weights='uniform', metric='euclidean') #instantiate model (i.e., choose model parameters)
kNN.fit(X_train_training, y_train_training)# fit model to data


# In[58]:


a=Model(kNN)
a.Dataset(X_test_testing, X_train_training)
a.Pred_accur()


# In[59]:


a.ConfusionMatrics()


# ### KNN (Standarize)

# In[60]:


kNN_param_grid = [  {'n_neighbors': range(1, 50)} ] # find the best k

kNN_grid_search_sc = GridSearchCV(KNeighborsClassifier(), kNN_param_grid, cv=20)


# In[61]:


kNN_grid_search_sc.fit(X_train_sc, y_train_training)


# In[62]:


print(f'The best parameters found are: {kNN_grid_search_sc.best_params_}')
print(f'The mean cross-validated accuracy is: {kNN_grid_search_sc.best_score_}')

print(f'predictions on the few examples under the best parameter setting: {kNN_grid_search_sc.predict(X_sc)[:40]}') 
print(f'accuracy of the best model on the whole dataset: {accuracy_score(y, kNN_grid_search_sc.predict(X_sc))}')


# In[63]:


kNN_sc = KNeighborsClassifier(n_neighbors=42) 
kNN_sc.fit(X_train_sc, y_train_training) 


# In[64]:


a=Model(kNN_sc)
a.Dataset(X_test_sc, X_train_sc)
a.Pred_accur()


# In[65]:


a.ConfusionMatrics()


# ### KNN (Normalize)

# In[66]:


kNN_param_grid = [  {'n_neighbors': range(1, 50)} ] # find the best k

kNN_grid_search_normali = GridSearchCV(KNeighborsClassifier(), kNN_param_grid, cv=20)


# In[67]:


kNN_grid_search_normali.fit(X_train_normali, y_train_training)


# In[68]:


print(f'The best parameters found are: {kNN_grid_search_normali.best_params_}')
print(f'The mean cross-validated accuracy is: {kNN_grid_search_normali.best_score_}')

print(f'predictions on the few examples under the best parameter setting: {kNN_grid_search_normali.predict(X_normali)[:40]}') 
print(f'accuracy of the best model on the whole dataset: {accuracy_score(y, kNN_grid_search_normali.predict(X_normali))}')


# In[69]:


kNN_normali = KNeighborsClassifier(n_neighbors=13) 
kNN_normali.fit(X_train_normali, y_train_training) 


# In[70]:


a=Model(kNN_normali)
a.Dataset(X_test_normali, X_train_normali)
a.Pred_accur()


# In[71]:


a.ConfusionMatrics()


# # Logistic-Regression

# In[72]:


log_reg = LogisticRegression(max_iter=300, random_state=1) # model instantiation. Keep in mind that if the class labels are not binary, this function will adopt the one-vs-all strategy.

from sklearn.model_selection import cross_val_score 

cross_val_res=cross_val_score(log_reg, X, y, cv=10) # do cross validation

print(np.mean(cross_val_res)) # find the cross-validated accuracy


# In[73]:


log_reg = LogisticRegression(max_iter=300, random_state=1) # model instantiation. Keep in mind that if the class labels are not binary, this function will adopt the one-vs-all strategy.

from sklearn.model_selection import cross_val_score 

cross_val_res_sc=cross_val_score(log_reg, X_sc, y, cv=10) # do cross validation

print(np.mean(cross_val_res_sc)) # find the cross-validated accuracy


# In[74]:


log_reg = LogisticRegression(max_iter=300, random_state=1) # model instantiation. Keep in mind that if the class labels are not binary, this function will adopt the one-vs-all strategy.

from sklearn.model_selection import cross_val_score 

cross_val_res_normali=cross_val_score(log_reg, X_normali, y, cv=10) # do cross validation

print(np.mean(cross_val_res_normali)) # find the cross-validated accuracy


#  # Figure generation

# In[75]:


# TODO: Your code for generating figures for the should be inserted here. Create more cells as you see fit. 
# Suggestion: compare the accuracies of the models you have built. 
class Compare: #Create a class Model to store it's Contrustor, maxium_Acc, Finalcompare, Finalcompare_sc, Finalcompare_normali
    def __init__(self, tree, NB, kNN):
        self.tree = tree #Create name to store tree classifier
        self.NB = NB #Create name to store NB classifier
        self.kNN = kNN #Create name to store kNN classifier
        
        self.pred_test1 = self.tree.predict(X_test_testing) # call the predict test value of tree
        self.pred_test2 = self.NB.predict(X_test_testing) # call the predict test value of NB
        self.pred_test3 = self.kNN.predict(X_test_testing) # call the predict test value of kNN
        
        self.pred_train1 = self.tree.predict(X_train_training) # call the predict train value of tree
        self.pred_train2 = self.NB.predict(X_train_training) # call the predict train value of NB
        self.pred_train3 = self.kNN.predict(X_train_training) # call the predict train value of kNN
        
        self.pred_test4 = self.tree.predict(X_test_sc)# call the predict test value of tree (Standarize)
        self.pred_test5 = self.NB.predict(X_test_sc) # call the predict test value of NB (Standarize)
        self.pred_test6 = self.kNN.predict(X_test_sc) # call the predict test value of kNN (Standarize)
        
        self.pred_train4 = self.tree.predict(X_train_sc) # call the predict train value of tree (Standarize)
        self.pred_train5 = self.NB.predict(X_train_sc)  # call the predict train value of NB (Standarize)
        self.pred_train6 = self.kNN.predict(X_train_sc) # call the predict train value of kNN (Standarize)
        
        self.pred_test7 = self.tree.predict(X_test_normali) # call the predict test value of tree (Normalization)
        self.pred_test8 = self.NB.predict(X_test_normali) # call the predict test value of NB (Normalization)
        self.pred_test9 = self.kNN.predict(X_test_normali) # call the predict test value of kNN (Normalization)
        
        self.pred_train7 = self.tree.predict(X_train_normali) # call the predict train value of tree (Normalization)
        self.pred_train8 = self.NB.predict(X_train_normali) # call the predict train value of NB (Normalization)
        self.pred_train9 = self.kNN.predict(X_train_normali)# call the predict train value of kNN (Normalization)
    
    def maximum_Acc(group):
        print('The Largest accuracies is: ', max(group.keys()), max(group.values()))# Find the maximum accuracy in different model
        
    def Finalcompare(self):
        print('Compare the accuracies of Train of 3 models (Descision Tree, Navie Bayes, kNN)')
        Group = {'Descision Tree': accuracy_score(y_train_training, self.pred_train1), 'Navie Bayes': accuracy_score(y_train_training, self.pred_train2),
                    'kNN': accuracy_score(y_train_training, self.pred_train3)}# Use Libiary to store classifier and accurate value
        for x, group in Group.items(): #Print them
            print(f"{x}: {group}")

        print('')

        print('Compare the accuracies of Test of 3 models (Descision Tree, Navie Bayes, kNN)')
        Group_test = {'Descision Tree': accuracy_score(y_test, self.pred_test1), 'Navie Bayes': accuracy_score(y_test, self.pred_test2),
                    'kNN': accuracy_score(y_test, self.pred_test3)}# Use Libiary to store classifier and accurate value
        for x, group in Group_test.items():#Print them
            print(f"{x}: {group}")
            
        Compare.maximum_Acc(Group_test)
    def Finalcompare_sc(self):
        print('Compare the accuracies of Train of 3 models (Descision Tree, Navie Bayes, kNN)')
        Group_sc = {'Descision Tree': accuracy_score(y_train_training, self.pred_train4), 'Navie Bayes': accuracy_score(y_train_training, self.pred_train5),
                    'kNN': accuracy_score(y_train_training, self.pred_train6)}# Use Libiary to store classifier and accurate value
        for x_sc, group_sc in Group_sc.items():#Print them
            print(f"{x_sc}: {group_sc}")

        print('')

        print('Compare the accuracies of Test of 3 models (Descision Tree, Navie Bayes, kNN)')
        Group_test_sc = {'Descision Tree': accuracy_score(y_test, self.pred_test4), 'Navie Bayes': accuracy_score(y_test, self.pred_test5),
                    'kNN': accuracy_score(y_test, self.pred_test6)}# Use Libiary to store classifier and accurate value
        for x_sc, group_sc in Group_test_sc.items():#Print them
            print(f"{x_sc}: {group_sc}")
        
        Compare.maximum_Acc(Group_test_sc)
    def Finalcompare_normali(self):
        print('Compare the accuracies of Train of 3 models (Descision Tree, Navie Bayes, kNN)')
        Group_normali = {'Descision Tree': accuracy_score(y_train_training, self.pred_train7), 'Navie Bayes': accuracy_score(y_train_training, self.pred_train8),
                    'kNN': accuracy_score(y_train_training, self.pred_train9)}# Use Libiary to store classifier and accurate value
        for x_normali, group_normali in Group_normali.items():#Print them
            print(f"{x_normali}: {group_normali}")

        print('')

        print('Compare the accuracies of Test of 3 models (Descision Tree, Navie Bayes, kNN)')
        Group_test_normali = {'Descision Tree': accuracy_score(y_test, self.pred_test7), 'Navie Bayes': accuracy_score(y_test, self.pred_test8),
                    'kNN': accuracy_score(y_test, self.pred_test9)}# Use Libiary to store classifier and accurate value
        for x_normali, group_normali in Group_test_normali.items():#Print them
            print(f"{x_normali}: {group_normali}")
        Compare.maximum_Acc(Group_test_normali)


# In[76]:


b=Compare(tree, NB, kNN) #Call the class Compare
b.Finalcompare() #Call the method FinalCompare


# In[77]:


b=Compare(tree, NB, kNN_sc)#Call the class Compare
b.Finalcompare_sc()#Call the method FinalCompare_sc


# In[78]:


b=Compare(tree, NB, kNN_normali)#Call the class Compare
b.Finalcompare_normali()#Call the method FinalCompare_normali


# In[79]:


print(np.mean(cross_val_score(tree, X, y, cv=5)))
print(np.mean(cross_val_score(NB, X, y, cv=5)))
print(np.mean(cross_val_score(kNN, X, y, cv=5)))
print(np.mean(cross_val_res))


# #### Generate y_test (kNN) and Final Decision

# In[80]:


Model(kNN)
a.Dataset(X_test_testing, X_train_training)
a.Pred_accur()


# In[81]:


Model(kNN)
a.Dataset(X_test, X_train)


# In[82]:


final=kNN_grid_search.predict(X_train)


# In[83]:


pd.DataFrame(final.shape)


# In[84]:


pd.DataFrame(final)[:10]


# I will choose kNN model and no processing dataset(X_train_traing, y_train_training) to be my final decision. Because it has high accuracy (around 70%), it has not imbalance data and it's test set is very close to train set.
# 
# Also, kNN model compare to other models such as decision tree and Naive Bayes, it has high accuary.
# 
# Meanwhile, no processing dataset is more accuarate than standarize and normalize datasets.
# 
# Therefore, kNN model and no processing dataset is the best result in this prediction.

#  # Exporting final prediction result

# In[85]:


def export_result_to_csv(predict, stu_name):
	"""[This functions export your prediction on X_test.csv]

	Args:
		predict (1d array): the ith entry gives the prediction of the ith test example. This format is identical to that of the predict() method of the models in sklearn. That is, you can given the output of the  predict() method as the parameter for the function.
		stu_name (string): your full English name
	"""
	pd.DataFrame(predict).to_csv(f"{stu_name}.csv", header=False, index=False) # name the output as your full English name

# you should call the above function to export your prediction result as a .csv file. Then, upload the file to Moodle. 
# NOTE: you should use this function only on your best model. 

# TODO: your code to call the export_result_to_csv function here....


# In[86]:


export_result_to_csv(final, "y_test")

