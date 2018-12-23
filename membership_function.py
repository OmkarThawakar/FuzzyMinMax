import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

gamma = 1

def func(r,gamma):
	if r*gamma>1:
		return 1
	elif 0<=r*gamma<=1 :
		return r*gamma
	else:
		return 0

def fuzzy_membership(x,v,w,gamma):

	xh_l , xh_u = x[0],x[1]
	return min([min(1-func(xh_u[i]-w[i] , gamma ) , 1-func(v[i]-xh_l[i] , gamma) ) for i in range(len(v))])

#print(fuzzy_membership([[0.2,0.2],[0.2,0.2]], [0.1,0.1] , [0.3,0.3] ,1))

'''
def fuzzy_membership(x,v,w,gamma):

	""""""
    return((sum([max(0,1-max(0,gamma*min(1,x[i]-w[i]))) for i in range(len(x))]) + \
          sum([max(0,1-max(0,gamma*min(1,v[i]-x[i]))) for i in range(len(x))]))/(2*len(x))
         )
'''
x,y = [i/50 for i in range(0,50)],[i/50 for i in range(0,50)]

points = []
for i in x:
    for j in y:
        points.append([i,j])

for i in range(len(points)):
    points[i].append(fuzzy_membership([points[i],points[i]],[0.4,0.2],[0.6,0.4],gamma))

points = np.array(points)

x,y,z = points[:,0],points[:,1],points[:,2]

X,Y = np.meshgrid(x,y)

Z = []
for i in range(len(z)):
    Z.append(z)
Z = np.array(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
ax.plot_trisurf(x, y, z,cmap='viridis', edgecolor='none');

ax.set_title('plot of Fuzzy membership function with gamma = {} '.format(gamma))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('membership value')
ax.view_init(60, 35)
plt.show()