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
result
![Alt text](https://github.com/OmkarThawakar/FuzzyMinMax/blob/master/FMM_Classification/result.jpg?raw=true "Hyperboxes")
