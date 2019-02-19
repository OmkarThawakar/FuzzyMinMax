# FuzzyMinMax
Fuzzy Min Max Neural Network Implementation

### Fuzzy Min Max Classification 

1. Import

```
from F_Min_Max import *
```

2. Create Network object

```
fuzzy = FuzzyMinMaxNN(1,theta=0.3) 
```

3. Create Dataset
```
X = [[0.2,0.2],[0.6,0.6],[0.5,0.5],[0.4,0.3],[0.8,0.1],[0.6,0.2],[0.7,0.6],[0.1,0.7],[0.3,0.9],[0.7,0.7],[0.9,0.9]]
d = [[1],[2],[1],[2],[1],[1],[2],[2],[2],[1],[1]]
```
4. Train the Network
```
fuzzy.train(X,d,1)
```
result
```

epoch : 1
==================================================
input pattern :  [0.2, 0.2] [1]
Hyperbox : [0.2, 0.2] , [0.4200108933349083, 0.5] 
Expanded Hyperbox :  [0.2, 0.2] [0.4200108933349083, 0.5]
Contracted  Hyperbox 1 :  [0.2, 0.2] [0.4200104353358848, 0.5]
Contracted Hyperbox 2 :  [0.42001020633637304, 0.3] [0.7, 0.6]
==================================================
input pattern :  [0.6, 0.6] [2]
Hyperbox : [0.42001020633637304, 0.3] , [0.7, 0.6] 
Expanded Hyperbox :  [0.42001020633637304, 0.3] [0.7, 0.6]
Contracted  Hyperbox 1 :  [0.2, 0.2] [0.4200103208361289, 0.5]
Contracted Hyperbox 2 :  [0.42001026358625093, 0.3] [0.7, 0.6]
==================================================
....
....
....

final hyperbox : 
V :  [[0.2, 0.2], [0.4200014053958256, 0.3], [0.6, 0.1], [0.1, 0.7], [0.7, 0.7]]
W :  [[0.4200023210014743, 0.5], [0.7, 0.6], [0.8, 0.2], [0.3, 0.9], [0.9, 0.9]]
```
5. Testing Pattern
```
fuzzy.predict([0.2,0.2])
```
result
```
pattern [0.2, 0.2] belongs to class 1 with fuzzy membership value : 1.0
pattern [0.2, 0.2] belongs to class 2 with fuzzy membership value : 0.9199996486510436
```
6. Visualizing Hyperboxes
```
fuzzy.show_hyperbox()
```
![Alt text](https://github.com/OmkarThawakar/FuzzyMinMax/blob/master/FMM_Classification/result.jpg?raw=true "Hyperboxes")

### Fuzzy Min Max Clustering 

1. Import

```
from F_Min_Max_Clustering import FuzzyMinMaxNN
```

2. Create Network object

```
fuzzy = FuzzyMinMaxNN(1,theta=0.4)  
```

3. Create Dataset
```
X = [[0.2,0.2],[0.6,0.6],[0.7,0.7],[0.4,0.7],[0.7,0.4]]
```
4. Train the Network
```
fuzzy.train(X,1)
```
result
```

epoch : 1
==================================================
input pattern :  [0.2, 0.2]
======================================================================
input pattern :  [0.6, 0.6]
Expanded Hyperbox :  [0.2, 0.2] [0.6, 0.6]
Hyperboxes : 
[[0.2, 0.2]]
[[0.6, 0.6]]
======================================================================
....
....
....

final hyperbox : 
V :  [[0.2, 0.2], [0.5, 0.5]]
W :  [[0.5, 0.5], [0.7, 0.7]]
======================================================================
```
5. Testing Pattern
```
fuzzy.predict([0.2,0.2])
```
result
```
pattern [0.2, 0.2] belongs to cluster 1 with fuzzy membership value : [1.0]
pattern [0.2, 0.2] belongs to cluster 2 with fuzzy membership value : [0.7]
```
6. Visualizing Hyperboxes
```
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax,a,b,label,color):
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
colors = ['g','r','cyan','k','yellow']

for i in range(len(fuzzy.V)):
    draw_box(ax,fuzzy.V[i],fuzzy.W[i],i+1,color=colors[i])

    
for i in range(len(X)):
    ax.scatter(X[i][0],X[i][1] , marker='*', c='k')

    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Fuzzy Min Max Clustering')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(('cluster 1','cluster 2'))
plt.show()
```
result
![Alt text](https://github.com/OmkarThawakar/FuzzyMinMax/blob/master/FMM_Clustering/result.jpg?raw=true "Hyperboxes")


### General Fuzzy Min Max 

it can be used for Classification and Clustering also.

1. Import

```
from GFMM import *
```

2. Create Network object

```
fuzzy = GFuzzyMinMaxNN(1,theta=0.3)  
```

3. Create Dataset
```
X = [[[0.4,0.3],[0.4,0.3]],[[0.6,0.25],[0.6,0.25]],[[0.7,0.2],[0.7,0.2]]]
d = [[1],[1],[0]]
```
4. Train the Network
```
fuzzy.train(X,d,1)
```
result
```
epoch : 1
======================================================================
input pattern :  [[0.4, 0.3], [0.4, 0.3]] [1]
Hyperbox : [0.4, 0.3] , [0.4, 0.3] 
======================================================================
input pattern :  [[0.6, 0.25], [0.6, 0.25]] [1]
Hyperbox : [0.4, 0.3] , [0.4, 0.3] 
Expanded Hyperbox :  [0.4, 0.25] [0.6, 0.3]
======================================================================
input pattern :  [[0.7, 0.2], [0.7, 0.2]] [0]
Hyperbox : [0.4, 0.25] , [0.6, 0.3] 
Expanded Hyperbox :  [0.4, 0.2] [0.7, 0.3]
======================================================================
final hyperbox : 
V :  [[0.4, 0.2]]
W :  [[0.7, 0.3]]
======================================================================
```
5. Testing Pattern
```
fuzzy.predict([[0.4, 0.3], [0.4, 0.3]])
```
result
```
pattern [[0.4, 0.3], [0.4, 0.3]] belongs to class 1 with fuzzy membership value : 1
pattern [[0.4, 0.3], [0.4, 0.3]] belongs to class 2 with fuzzy membership value : 0
```
6. Visualizing Hyperboxes
```
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
        
    elif d[i] == [1]:
        draw_box(ax,X[i][0],X[i][1],color='g')
        
    else:
        draw_box(ax,X[i][0],X[i][1],color='r')
        
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()
```
result
![Alt text](https://github.com/OmkarThawakar/FuzzyMinMax/blob/master/GFMM/result.png?raw=true "Hyperboxes")

### Enhanved Fuzzy Min Max Neural Network 

1. Import

```
from EFMM import FuzzyMinMaxNN
```

2. Create Network object

```
fuzzy = FuzzyMinMaxNN(1,theta=0.3)
```

3. Create Dataset
Here we are going to use Iris Dataset to train our Enhanced fuzzy min-max pattern classifier.
```
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',\
                   names=['PW','PL','SW','SL','Class'])
data['Class'] = data['Class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1,2,3])
np.random.shuffle(data.values)
data = data.sample(frac=1) #shuffle dataframe sample
df = data[['PW','PL','SW','SL']]
normalized_df=(df-df.min())/(df.max()-df.min())
#choose 50% training and 50% testing sample
train,test = normalized_df.values[:75,:4],normalized_df.values[75:,:4] 
train_labels,test_labels = data['Class'].values[:75],data['Class'].values[75:]
train_labels,test_labels = train_labels.reshape((-1,1)),test_labels.reshape((-1,1))

train,test = train.tolist(),test.tolist()
train_labels,test_labels = train_labels.tolist(),test_labels.tolist()


```
4. Train the Network
```
fuzzy.train(train,train_labels,1)
```
result
```
epoch : 1
======================================================================
input pattern :  [0.19444444444444448, 0.6666666666666666, 0.06779661016949151, 0.04166666666666667] [1]
Hyperbox : [0.19444444444444448, 0.6666666666666666, 0.06779661016949151, 0.04166666666666667] , [0.19444444444444448, 0.6666666666666666, 0.06779661016949151, 0.04166666666666667] 
======================================================================
input pattern :  [0.7222222222222222, 0.4583333333333333, 0.6610169491525424, 0.5833333333333334] [2]
Hyperbox : [0.7222222222222222, 0.4583333333333333, 0.6610169491525424, 0.5833333333333334] , [0.7222222222222222, 0.4583333333333333, 0.6610169491525424, 0.5833333333333334] 
======================================================================
....
....
....
final hyperbox : 
V :  [[0.19444444444444448, 0.5833333333333333, 0.07627118644067796, 0.08333333333333333], [0.6762152777777777, 0.3749999999999999,
....
....
0.6101694915254237, 0.5416666666666666], [0.42361111111111116, 0.20833333333333331, 0.652542372881356, 0.7083333333333334] ]
======================================================================
```

6. Visualizing Hyperboxes
Here our dataset has four dimension which unable to visualize. So we plot two dimension separately namely dim 1&2 and dim 3&4.
```
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

plt.show())
```
![Alt text](https://github.com/OmkarThawakar/FuzzyMinMax/blob/master/EFMM/iris_1_2.png?raw=true "Hyperboxes")

```
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
    
plt.xlabel('Dimension 3')
plt.ylabel('Dimension 4')
plt.title('Hyperboxes created during training')
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()

```
![Alt text](https://github.com/OmkarThawakar/FuzzyMinMax/blob/master/EFMM/iris_3_4.png?raw=true "Hyperboxes")


