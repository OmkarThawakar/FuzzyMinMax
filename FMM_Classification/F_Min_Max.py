
# coding: utf-8

# ### name : Omkar Thawakar
# #### Reg No : 2015BCS003  __ ____Roll_No : A-08 _____ Batch: A-01 
# #### Aim : Implement Fuzzy Min Max Neural Network Classifier 

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[6]:


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

                n_theta = sum([(max(self.W[i][_],x[_]) - min(self.V[i][_],x[_])) for _ in range(len(x))])
                
                if len(x)*self.theta >= n_theta :
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
        del_old = 1
        del_new = 1
        box_1,box_2,delta = -1,-1,-1
        for j in range(len(self.V)):
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
                           del_new = min(del_old,min(self.W[k][i]-self.V[j][i],self.W[j][i]-self.V[k][i]))
                    elif (self.V[k][i] < self.V[j][i] < self.W[j][i] < self.W[k][i]) :
                           del_new = min(del_old,min(self.W[j][i]-self.V[k][i],self.W[k][i]-self.V[j][i]))
                       
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
        if (self.V[box_1][delta] < self.V[box_2][delta] < self.W[box_1][delta] < self.W[box_2][delta]) :
            self.W[box_1][delta] = (self.W[box_1][delta]+self.V[box_2][delta])/2
            self.V[box_2][delta] = (self.W[box_1][delta]+self.V[box_2][delta])/2 
            
        elif (self.V[box_2][delta] < self.V[box_1][delta] < self.W[box_2][delta] < self.W[box_1][delta]) :
            self.W[box_2][delta] = (self.W[box_2][delta]+self.V[box_1][delta])/2
            self.V[box_1][delta] = (self.W[box_2][delta]+self.V[box_1][delta])/2
            
        elif (self.V[box_1][delta] < self.V[box_2][delta] < self.W[box_2][delta] < self.W[box_1][delta]) :
            if (self.W[box_2][delta]-self.V[box_1][delta])<(self.W[box_1][delta]-self.V[box_2][delta]):
                self.V[box_1][delta] = self.W[box_2][delta]
            else:
                self.W[box_1][delta] = self.V[box_2][delta]
            
        elif (self.V[box_2][delta] < self.V[box_1][delta] < self.W[box_1][delta] < self.W[box_2][delta]) :
            if (self.W[box_2][delta]-self.V[box_1][delta])<(self.W[box_1][delta]-self.V[box_2][delta]):
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
            print('='*50)

            for x,d in zip(X,d_):
                '''Get most sutaible hyperbox!!'''
                i , expand = self.get_hyperbox(x,d)

                print('input pattern : ',x , d)
                print('Hyperbox : {} , {} '.format(self.V[i],self.W[i])  )

                if expand:
                    self.expand(x,i)
                    print("Expanded Hyperbox : ",self.V[i] , self.W[i])
                    delta,j,k = self.overlap_Test()
                    if delta!=-1:
                        self.contraction(delta,j,k)
                        print("Contracted  Hyperbox 1 : ",self.V[j] , self.W[j])
                        print("Contracted Hyperbox 2 : ",self.V[k] , self.W[k])

                print('='*50)
            
        print('final hyperbox : ')
        print('V : ',self.V)
        print('W : ',self.W)
        
        
    def draw_box(self,ax,a,b,color):
        width = abs(a[0] - b[0])
        height = abs(a[1] - b[1])
        ax.add_patch(patches.Rectangle(a, width, height, fill=False,edgecolor=color))
        
    def show_hyperbox(self):
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
                self.draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='g')
            else:
                self.draw_box(ax,fuzzy.V[i],fuzzy.W[i],color='r')

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
        


# In[7]:


fuzzy = FuzzyMinMaxNN(1,theta=0.3)    


# ## Dataset

# In[8]:


X = [[0.2,0.2],[0.6,0.6],[0.5,0.5],[0.4,0.3],[0.8,0.1],[0.6,0.2],[0.7,0.6],[0.1,0.7],[0.3,0.9],[0.7,0.7],[0.9,0.9]]
d = [[1],[2],[1],[2],[1],[1],[2],[2],[2],[1],[1]]


# In[9]:


fuzzy.train(X,d,1)


# ### Testing of pattern 

# In[10]:


for x in X:
    fuzzy.predict(x)
    print('='*80)


# In[13]:


fuzzy.show_hyperbox()


# ### Visualization of HyperBoxes

# In[487]:


import matplotlib.pyplot as pltc
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

