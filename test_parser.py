import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from .Mixer import RequestMixer
from .Types import *


# data_path = "../Boulder_ConnectingRidge.csv"
# with open(data_path, "r") as f:
#     xy_data = np.loadtxt(f, delimiter=",", skiprows=1)
#     xy_data = xy_data[:, :2]
# np.save("rocks.npy", xy_data)

xy_data = np.random.uniform(-50, 50, size=(100, 2))

parser = Parser_T(randomization_space=2, data=xy_data)
parser_layer = Parser_Layer_T(output_space=2)

dem = np.zeros((4000, 4000), dtype=np.float32)
clipper = ImageClipper_T(randomization_space=1, mpp_resolution=1, data=dem)
clipper_layer = Image_T(output_space=1)

rpy_sampler = UniformSampler_T(randomization_space=4)
rpy = RollPitchYaw_T(output_space=4)

uni1 = UniformSampler_T(randomization_space=1)
line1 = Line_T(xmin=5.0, xmax=20.0)

req_xy = UserRequest_T(p_type = Position_T(), sampler=parser, layer=parser_layer, axes=["x","y"])
req_z = UserRequest_T(p_type=Position_T(), sampler=clipper, layer=clipper_layer, axes=["z"])
req_xyzw = UserRequest_T(p_type=Orientation_T(), sampler=rpy_sampler, layer=rpy, axes=["x", "y", "z", "w"])
req_scale = UserRequest_T(p_type = Scale_T(), sampler=uni1, layer=line1, axes=["xyz"])

request = [req_xy, req_z, req_xyzw, req_scale]
mixer = RequestMixer(request)

attributes = mixer.executeGraph(len(xy_data))
position = attributes["xformOp:translation"]
scale = attributes["xformOp:scale"]
orientation = attributes["xformOp:orientation"]

print('position: ', position.shape)
print('orientation: ', orientation.shape)
print('scale: ', scale.shape)

## Make sure that both plot are exactly same.
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xy_data[:, 0], xy_data[:, 1], 'ro')
plt.subplot(2, 1, 2)
plt.plot(position[:, 0], position[:, 1], 'bo')
plt.show()