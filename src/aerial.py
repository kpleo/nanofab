"""
Lithography solver

@author: Peng Kang
@email: kangp@physics.mcgill.ca
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy.integrate import cumtrapz
import skfmm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class AerialImage:
    """
    Class: Generate aerial image based upon given mask shape
    -------
    Input: Mask shape as numpy ndarray format.

    Output: Light intensity on the surface of photoresist according to mask shape

    """
    def __init__(self, mask, Lx=0.441, Ly=0.441):
        Nx = mask.shape[0]
        Ny = mask.shape[1]

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        grid_x = np.linspace(-(Nx - 1) / 2, (Nx - 1) / 2, Nx)
        grid_y = np.linspace(-(Ny - 1) / 2, (Ny - 1) / 2, Ny)

        nx, ny = np.meshgrid(grid_x, grid_y)
        fx = (1 / dx) * (1 / Nx) * nx
        fy = (1 / dx) * (1 / Ny) * ny

        self.mask = mask
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.fx = fx
        self.fy = fy


    def Mask_plot(self):
        plt.imshow(np.flipud(mask))
        plt.show()


    def AerialImg(self, NA=0.75, lamda=0.1):

        I_unshift = np.fft.fft2(self.mask)
        I = np.fft.fftshift(I_unshift)

        P = np.sqrt((self.fx ** 2) + (self.fy ** 2))
        freq = NA / lamda
        P = np.where(P < freq, 1, 0)
        I_f = np.fft.ifft2(P * I)
        I_f = np.real(I_f * np.conj(I_f))

        return I_f


    def AerialImg_plot(self):
        plt.pcolormesh(self.nx, self.ny, self.AerialImg())
        plt.show()


class Exposure:
    """

    """
    def __init__(self, I_f, nz=50, thickness=1000, alpha = 0.0005,
                 dill_a = 0.00075, dill_b = 0.00005, dill_c = 0.0025,
                 lamp_power = 30000, dose = 2000, n_steps = 50):
        self.I_f = I_f
        self.nz = nz
        self.thickness = thickness
        # alpha coefficient for beer lambert absorption
        self.alpha = alpha
        self.n_slice = I_f.shape[0]
        self.Nx = I_f.shape[1]
        # Instanciate resist parameters A in [1/nm], B in [1/nm] and C in [m²/J]
        self.dill_a = dill_a
        self.dill_b = dill_b
        self.dill_c = dill_c
        # Typical lamp power in W/m²
        self.lamp_power = lamp_power
        # Dose in J/m²
        self.dose = dose
        self.n_steps = n_steps

    def LatentImg(self, n=0):

        ######## Calculate 2d bulk image I(x,z) at n slice ##################
        grid_x = np.linspace(-(self.Nx - 1) / 2, (self.Nx - 1) / 2, self.Nx)
        # Compute the aerial image
        slice_xz = self.I_f[n,:]
        aerial_image_2d = slice_xz.tolist()
        # Create a meshgrid corresponding to the resist coordinates in x and z direction
        z = np.linspace(0, self.thickness, self.nz)
        grid_real = grid_x * 10
        X, Z = np.meshgrid(grid_real, z)
        # Instanciate bulk image, the aerial image is stacked with itself nz times.
        aerial_image = np.stack([aerial_image_2d for _ in range(self.nz)])
        bulk_ini = np.stack(aerial_image, 0)
        # Apply beer Lambert absorption
        bulk_img_slice = bulk_ini * np.exp(-self.alpha * Z)

        ######### Calculate latent image during exposition ##################
        # Initialise latent image
        lat_img = np.ones_like(bulk_img_slice)
        # Exposure time in s
        t_tot = self.dose / self.lamp_power

        # Discretize exposure time
        time_step = t_tot / self.n_steps
        # Loop to compute exposition
        for n in range(self.n_steps):
            # Latent image update
            lat_img *= np.exp(-self.dill_c * bulk_img_slice * time_step * self.lamp_power)
            # Absorption coefficient update
            alpha = self.dill_a * lat_img + self.dill_b
            # Bulk image update
            bulk_img_slice = bulk_ini * np.exp(-alpha * Z)

        lat_img_data = (lat_img, self.nz, self.thickness, X, Z)

        return lat_img_data


class Development:
    """

    """
    def __init__(self, lat_img_data, m_th=0.01, r_min=0.8, r_max=50, n_rate=2):
        self.lat_img = lat_img_data[0]
        # developement rate according to the 4 parameters model from Mack
        self.m_th = m_th
        self.r_min = r_min
        self.r_max = r_max
        self.n_rate = n_rate
        self.nz = lat_img_data[1]
        self.thickness = lat_img_data[2]
        self.X = lat_img_data[3]
        self.Z = lat_img_data[4]

    def Mack_Developement_Rate(self):
        a_mack = (1 - self.m_th) ** self.n_rate
        a_mack *= (self.n_rate + 1) / (self.n_rate - 1)
        dev_rate = (a_mack + 1) * (1 - self.lat_img) ** self.n_rate
        dev_rate /= a_mack + (1 - self.lat_img) ** self.n_rate
        dev_rate *= self.r_max
        dev_rate += self.r_min
        dev_rate = np.clip(dev_rate, self.r_min, self.r_max)

        # Computation of the development rate with typical parameters
        time_resist_z = cumtrapz(1. / dev_rate, dx=self.thickness/self.nz, axis=0, initial=0)
        cs = plt.contour(self.X, self.Z, time_resist_z, levels=[60, ])
        # Collect coutour data
        p = cs.collections[0].get_paths()
        x = np.array([]).reshape(1,0)
        y = np.array([]).reshape(1,0)

        for n in range(len(p)):
            v = p[n].vertices

            x = np.concatenate((x, np.array(v[:, 0]).reshape(1,-1)), axis=1)
            y = np.concatenate((y, np.array(v[:, 1]).reshape(1,-1)), axis=1)

        return x, y



if __name__ == '__main__':
    mask_file = "./Aerial_image/mask.mat"
    data = scio.loadmat(mask_file)
    mask = data["mask"]

    Obj_litho = AerialImage(mask)

    I_f = Obj_litho.AerialImg()

    Obj_expo = Exposure(I_f)

    N = 257
    grid = np.linspace(-(N- 1) / 2, (N - 1) / 2, N) * 10

    surface = np.zeros((257,2561))
    for n in range(257):
        lat_img_data = Obj_expo.LatentImg(n=n)
        Obj_dev = Development(lat_img_data)
        x, y = Obj_dev.Mack_Developement_Rate()
        # print(x.tolist())
        for a, data in enumerate(x.tolist()[0]):
            # print(data)
            surface[n,int(data+(2561-1)/2)] = y[:,a]
    plt.cla()
    # plt.imshow(surface)
    # plt.show()
    X = np.arange(-1280, 1281, 1)
    Y = np.arange(0, 257, 1)
    print(len(X),len(Y))
    XX, YY = np.meshgrid(X, Y)
    print(XX.shape, YY.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(XX, YY, surface, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.show()
