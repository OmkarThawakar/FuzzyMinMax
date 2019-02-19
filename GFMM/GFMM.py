
# coding: utf-8

# ### name : Omkar Thawakar
# #### Reg No : 2015BCS003  __ ____Roll_No : A-08 _____ Batch: A-01 
# #### Aim : Implement General Fuzzy Min Max Neural Network Classifier 

# In[1]:


import numpy as np
import pandas as pd


# In[84]:


class GFuzzyMinMaxNN:
    
    def __init__(self,sensitivity,theta=0.4):
        self.gamma = sensitivity
        self.hyperboxes = {}
        self.clasess = None
        self.V = []
        self.W = []
        self.U = []
        self.hyperbox_class = []
        self.theta = theta
        
        
    def func(self,r,gamma):
        if r*gamma>1:
            return 1
        elif 0<=r*gamma<=1 :
            return r*gamma
        else:
            return 0

    def fuzzy_membership(self,x,v,w):
        gamma = self.gamma
        xh_l , xh_u = x[0],x[1]
        return min([min(1-self.func(xh_u[i]-w[i] , gamma ) , 1-self.func(v[i]-xh_l[i] , gamma) ) for i in range(len(v))])

    
    def get_hyperbox(self,x,d):
        xh_l,xh_u = x[0],x[1]
        tmp = [0 for i in range(self.clasess)]
        tmp[d[0]-1] = 1
        
        """
            If no hyperbox present initially so create new
        """
        if len(self.V)==0 and len(self.W)==0 :
            self.V.append(xh_l)
            self.W.append(xh_u)
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
            if self.hyperbox_class[i]==[0]:
                test = [0 for _ in range(len(xh_l))]
                for _ in range(len(xh_l)):
                    if self.theta >= (max(self.W[i][_],xh_u[_])-min(self.V[i][_],xh_l[_])):
                        test[_]=1
                    else:
                        pass
                if test == [1 for _ in range(len(x))] :
                    expand = True
                    """
                        Label hyperbox with class label of input pattern
                    """
                    self.hyperbox_class[i][0]=d[0] 
                    return i,expand
            elif self.hyperbox_class[i]==d:
                mylist.append((self.fuzzy_membership(x,self.V[i],self.W[i])))
            else:
                mylist.append(-1)
                
        if len(mylist)>0:
            for box in sorted(mylist)[::-1]:
                i = mylist.index(box)
                test = [0 for i in range(len(xh_l))]
                for _ in range(len(xh_l)):
                    if self.theta >= (max(self.W[i][_],xh_u[_])-min(self.V[i][_],xh_l[_])):
                        test[_]=1
                    else:
                        pass

                if test == [1 for _ in range(len(x))] :
                    expand = True
                    return i,expand

            '''
                No hyperbox follow expansion criteria so create new
            '''
            self.V.append(xh_l)
            self.W.append(xh_u)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V)-1 , expand
            
        else:
            """
                If no hyperbox present for pattern x of class d so create new 
            """
            self.V.append(xh_l)
            self.W.append(xh_u)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V)-1,expand
    
    def expand(self,x,key):
        xh_l,xh_u = x[0],x[1]
        self.V[key] = [min(self.V[key][i],xh_l[i]) for i in range(len(xh_l))]
        self.W[key] = [max(self.W[key][i],xh_u[i]) for i in range(len(xh_u))]
        
    
    def overlap_Test(self):
        del_old = 1
        del_new = 1
        box_1,box_2,delta = -1,-1,-1
        for j in range(len(self.V)):
            if self.hyperbox_class[j] == [0] :
                for k in range(j+1,len(self.V)):
                    for i in range(len(self.V[j])):
                    
                        """
                            Test Four cases given by Patrick Simpson
                        """
                    
                        if (self.V[j][i] < self.V[k][i] < self.W[j][i] < self.W[k][i]) :
                               del_new = min(del_old,self.V[j][i]-self.V[k][i])
                        elif (self.V[k][i] < self.V[j][i] < self.W[k][i] < self.W[j][i]) :
                               del_new = min(del_old,self.W[k][i]-self.V[j][i])
                        elif (self.V[j][i] < self.V[k][i] < self.W[k][i] < self.W[j][i]) :
                               del_new = min(del_old,min(self.W[k][i]-self.V[j][i],                                                     self.W[j][i]-self.V[k][i]))
                        elif (self.V[k][i] < self.V[j][i] < self.W[j][i] < self.W[k][i]) :
                               del_new = min(del_old,min(self.W[j][i]-self.V[k][i],                                                     self.W[k][i]-self.V[j][i]))
                       
                        """
                            Check dimension for which overlap is minimum
                        """
                        #print(del_old , del_new , del_old-del_new , i)
                        if del_old - del_new > 0.0 :
                            delta = i
                            box_1,box_2 = j,k
                            del_old = del_new
                            del_new = 1
                        else:
                            pass
                        
            else:
                for k in range(j+1,len(self.V)):
                    if self.hyperbox_class[j]==self.hyperbox_class[k] :
                        pass
                    else:
                        for i in range(len(self.V[j])):

                            """
                                Test Four cases given by Patrick Simpson
                            """

                            if (self.V[j][i] < self.V[k][i] < self.W[j][i] < self.W[k][i]) :
                                   del_new = min(del_old,self.V[j][i]-self.V[k][i])
                            elif (self.V[k][i] < self.V[j][i] < self.W[k][i] < self.W[j][i]) :
                                   del_new = min(del_old,self.W[k][i]-self.V[j][i])
                            elif (self.V[j][i] < self.V[k][i] < self.W[k][i] < self.W[j][i]) :
                                   del_new = min(del_old,min(self.W[k][i]-self.V[j][i],                                                         self.W[j][i]-self.V[k][i]))
                            elif (self.V[k][i] < self.V[j][i] < self.W[j][i] < self.W[k][i]) :
                                   del_new = min(del_old,min(self.W[j][i]-self.V[k][i],                                                         self.W[k][i]-self.V[j][i]))

                            """
                                Check dimension for which overlap is minimum
                            """
                            #print(del_old , del_new , del_old-del_new , i)
                            if del_old - del_new > 0.0 :
                                delta = i
                                box_1,box_2 = j,k
                                del_old = del_new
                                del_new = 1
                            else:
                                pass
                
        return delta , box_1, box_2
            
                       
    def contraction(self,delta,box_1,box_2):
        if (self.V[box_1][delta] < self.V[box_2][delta] < self.W[box_1][delta] <             self.W[box_2][delta]) :
            self.W[box_1][delta] = (self.W[box_1][delta]+self.V[box_2][delta])/2
            self.V[box_2][delta] = (self.W[box_1][delta]+self.V[box_2][delta])/2 
            
        elif (self.V[box_2][delta] < self.V[box_1][delta] < self.W[box_2][delta] <               self.W[box_1][delta]) :
            self.W[box_2][delta] = (self.W[box_2][delta]+self.V[box_1][delta])/2
            self.V[box_1][delta] = (self.W[box_2][delta]+self.V[box_1][delta])/2
            
        elif (self.V[box_1][delta] < self.V[box_2][delta] < self.W[box_2][delta] <               self.W[box_1][delta]) :
            if (self.W[box_2][delta]-self.V[box_1][delta])<             (self.W[box_1][delta]-self.V[box_2][delta]):
                self.V[box_1][delta] = self.W[box_2][delta]
            else:
                self.W[box_1][delta] = self.V[box_2][delta]
            
        elif (self.V[box_2][delta] < self.V[box_1][delta] < self.W[box_1][delta] <               self.W[box_2][delta]) :
            if (self.W[box_2][delta]-self.V[box_1][delta])<            (self.W[box_1][delta]-self.V[box_2][delta]):
                self.W[box_2][delta] = self.V[box_1][delta]
            else:
                self.V[box_2][delta] = self.W[box_1][delta]
    
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
                     to cases given by Gabrys and Bargiela
                    """
                    self.overlap_Test()
                    

                print('='*70)
            
        print('final hyperbox : ')
        print('V : ',self.V)
        print('W : ',self.W)
        


# In[51]:


fuzzy = GFuzzyMinMaxNN(1,theta=0.3)    


# ## Dataset

# In[52]:


fuzzy.fuzzy_membership([[0.4,0.3],[0.4,0.3]],[0.4,0.3],[0.4,0.3])


# In[53]:


X = [[[0.4,0.3],[0.4,0.3]],[[0.6,0.25],[0.6,0.25]],[[0.7,0.2],[0.7,0.2]]]
d = [[1],[1],[0]]


# In[54]:


fuzzy.train(X,d,1)


# ### Testing of pattern 

# In[55]:


for x in X:
    fuzzy.predict(x)
    print('='*80)


# ### Visualization of HyperBoxes

# In[60]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,color):
    if a==b :
        ax.scatter(a[0],a[1] , marker='*', c=color)
    else:
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
    if d[i] == [0]:
        draw_box(ax,X[i][0],X[i][1],color='r')
        #ax.scatter(X[i][0],X[i][1] , marker='o', c='g')
    elif d[i] == [1]:
        draw_box(ax,X[i][0],X[i][1],color='g')
        #ax.scatter(X[i][0],X[i][1] , marker='o', c='g')
    else:
        draw_box(ax,X[i][0],X[i][1],color='r')
        #ax.scatter(X[i][0],X[i][1] , marker='*', c='r')
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])
#plt.legend(('class 1','class 2'))
plt.show()


# In[59]:


X[0][0] , fuzzy.V[0]


# ## Iris Dataset Classification using Fuzzy Min Max NN

# In[62]:


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',                   names=['PW','PL','SW','SL','Class'])


# In[63]:


data['Class'] = data['Class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1,2,3])


# In[64]:


np.random.shuffle(data.values)


# In[65]:


data = data.sample(frac=1) #shuffle dataframe sample


# In[66]:


data.head()


# In[67]:


df = data[['PW','PL','SW','SL']]


# In[68]:


df.head()


# In[69]:


normalized_df=(df-df.min())/(df.max()-df.min())


# In[70]:


data['Class'].values


# In[71]:


#choose 50% training and 50% testing sample
train,test = normalized_df.values[:75,:4],normalized_df.values[75:,:4] 
train_labels,test_labels = data['Class'].values[:75],data['Class'].values[75:]
train_labels,test_labels = train_labels.reshape((-1,1)),test_labels.reshape((-1,1))


# In[72]:


train.shape,test.shape


# In[73]:


train_labels.shape,test_labels.shape


# In[74]:


train,test = train.tolist(),test.tolist()
train_labels,test_labels = train_labels.tolist(),test_labels.tolist()


# In[76]:


for i in range(len(train)):
    train[i] = [train[i],train[i]]
for i in range(len(test)):
    test[i] = [test[i],test[i]]


# In[77]:


train[0]


# In[95]:


fuzzy = GFuzzyMinMaxNN(1,theta=0.5)


# In[96]:


fuzzy.train(train,train_labels,1)


# In[97]:


len(fuzzy.V),len(fuzzy.W)


# In[101]:


fuzzy.V[74] , fuzzy.W[74]


# ## In dimension 1 & 2

# In[98]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,color):
    if a==b :
        ax.scatter(a[0],a[1] , marker='*', c=color)
    else:
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
        draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='r')
    elif fuzzy.hyperbox_class[i]==[2]:
        draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='g')
    else:
        draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='b')
    
for i in range(len(train)):
    if train_labels[i] == [1]:
        draw_box(ax,train[i][0],train[i][1],color='r')
        
    elif train_labels[i] == [2]:
        draw_box(ax,train[i][0],train[i][1],color='g')
        
    else:
        draw_box(ax,train[i][0],train[i][1],color='b')
        
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])
#plt.legend(('class 1','class 2'))
plt.show()


# In[102]:


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
      


# In[103]:


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


# In[104]:


print('Accuracy (train) : {} %'.format(score(train,train_labels)))


# In[105]:


print('Accuracy (test) : {} %'.format(score(test,test_labels)))

