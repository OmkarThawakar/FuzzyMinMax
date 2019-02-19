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

