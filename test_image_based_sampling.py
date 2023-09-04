import numpy as np
import dataclasses
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from .Types import *
from .Layers import *
from .Samplers import *

trans = Translation3D_T(z=0.25)
quat =Quaternion_T(x=0.383,y=0.0,z=0.0,w=0.924)
T = Transformation3D_T(translation=trans, orientation=quat)

uni2 = UniformSampler_T(randomization_space=2)
plane = Plane_T(xmin=-0.5,xmax=0.5,ymin=-0.5,ymax=0.5, transform=T, output_space=3)

import cv2
img = np.zeros((101,101),dtype=np.uint8)
img = cv2.circle(img, (50,50), 25, 1, -1)
img = cv2.circle(img, (50,50), 5, 0, -1)
img2 = np.zeros((998,652),dtype=np.uint8)
img2 = cv2.circle(img2, (300,550), 200, 1, -1)
img2 = cv2.circle(img2, (200,550), 200, 0, -1)
cv2.putText(img2, 'SpaceR', (80,250), cv2.FONT_HERSHEY_SIMPLEX, 4, 1, 8, cv2.LINE_AA)
img2 = np.flip(img2,1)
np.save("mask.npy",img2)
plt.figure()
plt.imshow(img2)
plt.show()

image = Image_T(data=img, mpp_resolution=0.01, transform=T, output_space=3)
image2 = Image_T(data=img2, mpp_resolution=0.01, output_space=2)


cube_layer = PlaneLayer(plane, uni2)
points = cube_layer(1000)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Uniform on a Plane")

cube_layer = ImageLayer(image, uni2)
points = cube_layer(1000)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],'+')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_title("Uniform on a Plane")

cube_layer = ImageLayer(image2, uni2)
points = cube_layer(1000)
print(points.shape)
print(points[:,0].max())
print(points[:,1].max())
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(points[:,0],points[:,1],0,'+')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_title("Uniform on an Image")

plt.show()