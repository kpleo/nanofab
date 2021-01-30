import scipy.io as scio
from matplotlib import cm
from src.aerial import AerialImage,Exposure,Development
import numpy as np
import matplotlib.pyplot as plt


mask_file = "./Aerial_image/mask.mat"
data = scio.loadmat(mask_file)
mask = data["mask"]

Obj_litho = AerialImage(mask)

I_f = Obj_litho.AerialImg()

Obj_expo = Exposure(I_f)

N = 257
grid = np.linspace(-(N - 1) / 2, (N - 1) / 2, N) * 10

surface = np.zeros((257, 2561))
for n in range(257):
    lat_img_data = Obj_expo.LatentImg(n=n)
    Obj_dev = Development(lat_img_data)
    x, y = Obj_dev.Mack_Developement_Rate()
    # print(x.tolist())
    for a, data in enumerate(x.tolist()[0]):
        # print(data)
        surface[n, int(data + (2561 - 1) / 2)] = y[:, a]
plt.cla()
# plt.imshow(surface)
# plt.show()
X = np.arange(-1280, 1281, 1)
Y = np.arange(0, 257, 1)
print(len(X), len(Y))
XX, YY = np.meshgrid(X, Y)
print(XX.shape, YY.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX, YY, surface, cmap=cm.coolwarm, linewidth=0, antialiased=True)
plt.show()

scio.savemat("./surface.mat", {"surface": surface})