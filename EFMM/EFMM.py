
# coding: utf-8

# ### name : Omkar Thawakar
# #### Reg No : 2015BCS003  __ ____Roll_No : A-08 _____ Batch: A-01 
# #### Aim : Implement Enhanced Fuzzy Min Max Neural Network Classifier 

# In[1]:


import numpy as np
import pandas as pd


# In[85]:


class FuzzyMinMaxNN:
    
    def __init__(self,sensitivity,theta=0.4):
        self.gamma = sensitivity
        self.hyperboxes = {}
        self.clasess = None
        self.V = []
        self.W = []
        self.U = []
        self.hyperbox_class = []
        self.theta = theta
        
        
    def fuzzy_membership(self,x,v,w,gamma=1):
        """
            returns the fuzzy menbership function :
            b_i(xh,v,w) = 1/2n*(------)
        """
        return((sum([max(0,1-max(0,gamma*min(1,x[i]-w[i]))) for i in range(len(x))]) +           sum([max(0,1-max(0,gamma*min(1,v[i]-x[i]))) for i in range(len(x))]))/(2*len(x))
         )
    
    def get_hyperbox(self,x,d):
        
        tmp = [0 for i in range(self.clasess)]
        tmp[d[0]-1] = 1
        
        """
            If no hyperbox present initially so create new
        """
        if len(self.V)==0 and len(self.W)==0 :
            self.V.append(x)
            self.W.append(x)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V)-1 , expand 
                
        """
            returns the most sutaible hyperbox for input pattern x
            otherwise None
        """
        mylist = []
        
        for i in range(len(self.V)):
            if self.hyperbox_class[i]==d:
                mylist.append((self.fuzzy_membership(x,self.V[i],self.W[i])))
            else:
                mylist.append(-1)
                
        if len(mylist)>0:
            for box in sorted(mylist)[::-1]:
                i = mylist.index(box)
                
                test = [0 for i in range(len(x))]
                for _ in range(len(x)):
                    if self.theta >= (max(self.W[i][_],x[_])-min(self.V[i][_],x[_])):
                        test[_]=1
                    else:
                        pass

                if test == [1 for _ in range(len(x))] :
                    expand = True
                    return i,expand

            '''
                No hyperbox follow expansion criteria so create new
            '''
            self.V.append(x)
            self.W.append(x)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V)-1 , expand
            
        else:
            """
                If no hyperbox present for pattern x of class d so create new 
            """
            self.V.append(x)
            self.W.append(x)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V)-1,expand
    
    def expand(self,x,key):
        self.V[key] = [min(self.V[key][i],x[i]) for i in range(len(x))]
        self.W[key] = [max(self.W[key][i],x[i]) for i in range(len(x))]
        
    
    def overlap_Test(self):
        
        for j in range(len(self.V)):
            for k in range(j+1,len(self.V)):
                del_old = 1
                delta = -1
                for i in range(len(self.V[j])):
                    del_new = 1
                    
                    """
                        Test nine cases given by Mohammad Falah Mohammad
                        and contract hyperbox if needed
                    """
                    #case 1
                    if (self.V[j][i] < self.V[k][i] < self.W[j][i] < self.W[k][i]) :
                        del_new = min(del_old,self.W[j][i]-self.V[k][i]) 
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                            
                    #case 2    
                    elif (self.V[k][i] < self.V[j][i] < self.W[k][i] < self.W[j][i]) :
                        del_new = min(del_old,self.W[k][i]-self.V[j][i])
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                    #case 3    
                    elif (self.V[j][i] == self.V[k][i] < self.W[j][i] < self.W[k][i]) : #new case
                        del_new = min(del_old,min(self.W[j][i]-self.V[k][i],self.W[k][i]-self.V[j][i]))
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                            
                    #case 4       
                    elif (self.V[j][i] < self.V[k][i] < self.W[j][i] == self.W[k][i]) : #new case
                        del_new = min(del_old,min(self.W[j][i]-self.V[k][i],self.W[k][i]-self.V[j][i]))
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                    #case 5        
                    elif (self.V[k][i] == self.V[j][i] < self.W[k][i] < self.W[j][i]) : #new case
                        del_new = min(del_old,min(self.W[j][i]-self.V[k][i],self.W[k][i]-self.V[j][i]))
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                    #case 6       
                    elif (self.V[k][i] < self.V[j][i] < self.W[k][i] == self.W[j][i]) : #new case
                        del_new = min(del_old,min(self.W[j][i]-self.V[k][i],self.W[k][i]-self.V[j][i]))
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                    #case 7       
                    elif (self.V[j][i] < self.V[k][i] <= self.W[k][i] < self.W[j][i]) :
                        del_new = min(del_old,min(self.W[k][i]-self.V[j][i],self.W[j][i]-self.V[k][i]))
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                    #case 8        
                    elif (self.V[k][i] < self.V[j][i] <= self.W[j][i] < self.W[k][i]) :
                        del_new = min(del_old,min(self.W[j][i]-self.V[k][i],self.W[k][i]-self.V[j][i]))
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                    #case 9       
                    elif (self.V[k][i] == self.V[j][i] < self.W[k][i] == self.W[j][i]) : #new case
                        del_new = min(del_old,self.W[k][i]-self.V[j][i])
                        if del_old-del_new<1 :
                            delta = i
                            del_old = del_new
                            self.contraction(delta,j,k)
                        else:
                            pass
                        
                       
    def contraction(self,delta,j,k):
        #case 1
        if (self.V[j][delta] < self.V[k][delta] < self.W[j][delta] < self.W[k][delta]) :
            self.W[j][delta] = self.V[k][delta]=(self.W[j][delta]+self.V[k][delta])/2

        #case 2    
        elif (self.V[k][delta] < self.V[j][delta] < self.W[k][delta] < self.W[j][delta]) :
            self.W[k][delta] = self.V[j][delta]=(self.W[k][delta]+self.V[j][delta])/2

        #case 3    
        elif (self.V[j][delta] == self.V[k][delta] < self.W[j][delta] < self.W[k][delta]) : #new case
            self.V[k][delta] = self.W[j][delta]

        #case 4       
        elif (self.V[j][delta] < self.V[k][delta] < self.W[j][delta] == self.W[k][delta]) : #new case
            self.W[j][delta] = self.V[k][delta]

        #case 5        
        elif (self.V[k][delta] == self.V[j][delta] < self.W[k][delta] < self.W[j][delta]) : #new case
            self.V[j][delta] = self.W[k][delta]

        #case 6       
        elif (self.V[k][delta] < self.V[j][delta] < self.W[k][delta] == self.W[j][delta]) : #new case
            self.W[k][delta] = self.V[j][delta]

        #case 7       
        elif (self.V[j][delta] < self.V[k][delta] <= self.W[k][delta] < self.W[j][delta]) :
            if (self.W[k][delta]-self.V[j][delta]) < (self.W[j][delta]-self.V[k][delta]) :
                self.V[j][delta] = self.W[k][delta]
            else:
                self.W[j][delta] = self.V[k][delta]

        #case 8        
        elif (self.V[k][delta] < self.V[j][delta] <= self.W[j][delta] < self.W[k][delta]) :
            if (self.W[k][delta]-self.V[j][delta]) > (self.W[j][delta]-self.V[k][delta]) :
                self.W[k][delta] = self.V[j][delta]
            else:
                self.V[k][delta] = self.W[j][delta]

        #case 9 a      
        elif (self.V[j][delta] == self.V[k][delta] < self.W[j][delta] == self.W[j][delta]) : #new case
            self.W[j][delta] = self.V[k][delta] = (self.W[j][delta]+self.V[k][delta])/2
        #case 9 b
        elif (self.V[k][delta] == self.V[j][delta] < self.W[k][delta] == self.W[j][delta]) :
            self.W[k][delta] = self.V[j][delta] = (self.W[k][delta]+self.V[j][delta])/2

        print('Contracted hyperbox 1 : ',self.V[j],self.W[j])
        print('Contracted hyperbox 2 : ',self.V[k],self.W[k])
    
    def predict(self,x):
        mylist = []
        for i in range(len(self.V)):
            mylist.append([self.fuzzy_membership(x,self.V[i],self.W[i])])
            
        result = np.multiply(mylist,self.U)
        for i in range(self.clasess):
            print('pattern {} belongs to class {} with fuzzy membership value : {}'.format(x,i+1,max(result[:,i])))
        
            
                    
    def train(self,X,d_,epochs):
        self.clasess = len(np.unique(np.array(d_)))
        for _ in range(epochs):
            print('epoch : {}'.format(_+1))
            print('='*70)

            for x,d in zip(X,d_):
                '''Get most sutaible hyperbox!!'''
                i , expand = self.get_hyperbox(x,d)

                print('input pattern : ',x , d)
                print('Hyperbox : {} , {} '.format(self.V[i],self.W[i])  )

                if expand:
                    self.expand(x,i)
                    print("Expanded Hyperbox : ",self.V[i] , self.W[i])
                    
                    """
                     As hyperbox expanded cheak overlap and contract according 
                     to cases given by Mohammad Falah Mohammad
                    """
                    self.overlap_Test()
                    

                print('='*70)
            
        print('final hyperbox : ')
        print('V : ',self.V)
        print('W : ',self.W)
        


# In[86]:


fuzzy = FuzzyMinMaxNN(1,theta=0.3)    


# ## Dataset

# In[82]:


# X = [[0.2,0.2],[0.6,0.6],[0.5,0.5],[0.4,0.3],[0.8,0.1],[0.6,0.2],[0.7,0.6],[0.1,0.7],[0.3,0.9],[0.7,0.7],[0.9,0.9]]
# d = [[1],[2],[1],[2],[1],[1],[2],[2],[2],[1],[1]]


# In[87]:


X = [[0.4,0.3],[0.6,0.25],[0.7,0.2]]
d = [[1],[1],[1]]


# In[88]:


fuzzy.train(X,d,1)


# ### Testing of pattern 

# In[89]:


for x in X:
    fuzzy.predict(x)
    print('='*80)


# ### Visualization of HyperBoxes

# In[90]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,color):
    width = abs(a[0] - b[0])
    height = abs(a[1] - b[1])
    ax.add_patch(patches.Rectangle(a, width, height, fill=False,edgecolor=color))

"""
    plot dataset
"""
fig1 = plt.figure()
ax = fig1.add_subplot(111, aspect='equal',alpha=0.7)

        
"""
    plot Hyperboxes
"""
for i in range(len(fuzzy.V)):
    if fuzzy.hyperbox_class[i]==[1]:
        draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='g')
    else:
        draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='r')
    
for i in range(len(X)):
    if d[i] == [1]:
        ax.scatter(X[i][0],X[i][1] , marker='o', c='g')
    else:
        ax.scatter(X[i][0],X[i][1] , marker='*', c='r')
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])
#plt.legend(('class 1','class 2'))
plt.show()


# ## Iris Dataset Classification using Fuzzy Min Max NN

# In[91]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',                   names=['PW','PL','SW','SL','Class'])


# In[92]:


data['Class'] = data['Class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1,2,3])


# In[93]:


np.random.shuffle(data.values)


# In[94]:


data = data.sample(frac=1) #shuffle dataframe sample


# In[95]:


data.head()


# In[96]:


df = data[['PW','PL','SW','SL']]


# In[97]:


df.head()


# In[98]:


normalized_df=(df-df.min())/(df.max()-df.min())


# In[99]:


data['Class'].values


# In[100]:


#choose 50% training and 50% testing sample
train,test = normalized_df.values[:75,:4],normalized_df.values[75:,:4] 
train_labels,test_labels = data['Class'].values[:75],data['Class'].values[75:]
train_labels,test_labels = train_labels.reshape((-1,1)),test_labels.reshape((-1,1))


# In[101]:


train.shape,test.shape


# In[102]:


train_labels.shape,test_labels.shape


# In[103]:


train,test = train.tolist(),test.tolist()
train_labels,test_labels = train_labels.tolist(),test_labels.tolist()


# In[104]:


fuzzy = FuzzyMinMaxNN(1,theta=0.2)


# In[105]:


fuzzy.train(train,train_labels,1)


# In[106]:


len(fuzzy.V),len(fuzzy.W)


# ## In dimension 1 & 2

# In[107]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,color):
    width = abs(a[0] - b[0])
    height = abs(a[1] - b[1])
    ax.add_patch(patches.Rectangle(a, width, height, fill=False,edgecolor=color))

"""
    plot dataset
"""
fig1 = plt.figure()
ax = fig1.add_subplot(111, aspect='equal',alpha=0.7)

        
"""
    plot Hyperboxes
"""
for i in range(len(fuzzy.V)):
    if fuzzy.hyperbox_class[i]==[1]:
        draw_box(ax,fuzzy.V[i][:2],fuzzy.W[i][:2],color='g')
    elif fuzzy.hyperbox_class[i]==[2]:
        draw_box(ax,fuzzy.V[i][:2],fuzzy.W[i][:2],color='b')
    else:
        draw_box(ax,fuzzy.V[i][:2],fuzzy.W[i][:2],color='r')
    
for i in range(len(train)):
    if train_labels[i] == [1]:
        ax.scatter(train[i][0],train[i][1] , marker='o', c='g')
    elif train_labels[i] == [2]:
        ax.scatter(train[i][0],train[i][1] , marker='o', c='b')
    else:
        ax.scatter(train[i][0],train[i][1] , marker='o', c='r')
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()


# ## In Dimension 3 & 4

# In[108]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,color):
    width = abs(a[0] - b[0])
    height = abs(a[1] - b[1])
    ax.add_patch(patches.Rectangle(a, width, height, fill=False,edgecolor=color))

"""
    plot dataset
"""
fig1 = plt.figure()
ax = fig1.add_subplot(111, aspect='equal',alpha=0.7)

        
"""
    plot Hyperboxes
"""
for i in range(len(fuzzy.V)):
    if fuzzy.hyperbox_class[i]==[1]:
        draw_box(ax,fuzzy.V[i][2:],fuzzy.W[i][2:],color='g')
    elif fuzzy.hyperbox_class[i]==[2]:
        draw_box(ax,fuzzy.V[i][2:],fuzzy.W[i][2:],color='b')
    else:
        draw_box(ax,fuzzy.V[i][2:],fuzzy.W[i][2:],color='r')
    
for i in range(len(train)):
    if train_labels[i] == [1]:
        ax.scatter(train[i][2],train[i][3] , marker='o', c='g')
    elif train_labels[i] == [2]:
        ax.scatter(train[i][2],train[i][3] , marker='o', c='b')
    else:
        ax.scatter(train[i][2],train[i][3] , marker='o', c='r')
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()


# In[114]:


def get_class(x):
        mylist = []
        for i in range(len(fuzzy.V)):
            mylist.append([fuzzy.fuzzy_membership(x,fuzzy.V[i],fuzzy.W[i])])
        result = np.multiply(mylist,fuzzy.U)
        mylist=[]
        for i in range(fuzzy.clasess):
            mylist.append(max(result[:,i]))
            
        #print(mylist)
        #print(mylist.index(max(mylist))+1,max(mylist))
        #print('pattern belongs to class {} with fuzzy membership : {}'.format(mylist.index(max(mylist))+1,max(mylist)))
        return [mylist.index(max(mylist))+1]
      


# In[115]:


def score(train,train_labels):
    counter=0
    wronge=0
    for i in range(len(train)):
        if get_class(train[i]) == train_labels[i] :
            counter+=1
        else:
            wronge+=1
            
    print('No of misclassification : {}'.format(wronge))
    return (counter/len(train_labels))*100


# In[117]:


print('Accuracy (train) : {} %'.format(score(train,train_labels)))


# In[118]:


print('Accuracy (test) : {} %'.format(score(test,test_labels)))

