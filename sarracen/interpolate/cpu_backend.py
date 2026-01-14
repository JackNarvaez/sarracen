from typing import Tuple

from numba import njit, prange, get_num_threads
from numba.core.registry import CPUDispatcher
from numpy import ndarray
import numpy as np

from ..interpolate.base_backend import BaseBackend
from ..kernels.cubic_spline_exact import line_int, surface_int


class CPUBackend(BaseBackend):

    @staticmethod
    def interpolate_2d_render(x: ndarray,
                              y: ndarray,
                              weight: ndarray,
                              h: ndarray,
                              weight_function: CPUDispatcher,
                              kernel_radius: float,
                              x_pixels: int,
                              y_pixels: int,
                              x_min: float,
                              x_max: float,
                              y_min: float,
                              y_max: float,
                              exact: bool) -> ndarray:
        if exact:
            return CPUBackend._exact_2d_render(x, y, weight, h, x_pixels,
                                               y_pixels, x_min, x_max,
                                               y_min, y_max)
        return CPUBackend._fast_2d(x, y, np.zeros(x.size), 0, weight, h,
                                   weight_function, kernel_radius, x_pixels,
                                   y_pixels, x_min, x_max, y_min, y_max, 2)

    @staticmethod
    def interpolate_2d_render_vec(x: ndarray,
                                  y: ndarray,
                                  weight_x: ndarray,
                                  weight_y: ndarray,
                                  h: ndarray,
                                  weight_function: CPUDispatcher,
                                  kernel_radius: float,
                                  x_pixels: int,
                                  y_pixels: int,
                                  x_min: float,
                                  x_max: float,
                                  y_min: float,
                                  y_max: float,
                                  exact: bool) -> Tuple[ndarray, ndarray]:
        if exact:
            return (CPUBackend._exact_2d_render(x, y, weight_x, h, x_pixels,
                                                y_pixels, x_min, x_max,
                                                y_min, y_max),
                    CPUBackend._exact_2d_render(x, y, weight_y, h, x_pixels,
                                                y_pixels, x_min, x_max,
                                                y_min, y_max))
        return (CPUBackend._fast_2d(x, y, np.zeros(x.size), 0, weight_x, h,
                                    weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 2),
                CPUBackend._fast_2d(x, y, np.zeros(x.size), 0, weight_y, h,
                                    weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 2))

    @staticmethod
    def interpolate_2d_line(x: ndarray,
                            y: ndarray,
                            weight: ndarray,
                            h: ndarray,
                            weight_function: CPUDispatcher,
                            kernel_radius: float,
                            pixels: int,
                            x1: float,
                            x2: float,
                            y1: float,
                            y2: float) -> ndarray:
        return CPUBackend._fast_2d_line(x, y, weight, h, weight_function,
                                        kernel_radius, pixels, x1, x2,
                                        y1, y2)

    @staticmethod
    def interpolate_3d_line(x: ndarray,
                            y: ndarray,
                            z: ndarray,
                            weight: ndarray,
                            h: ndarray,
                            weight_function: CPUDispatcher,
                            kernel_radius: float,
                            pixels: int,
                            x1: float,
                            x2: float,
                            y1: float,
                            y2: float,
                            z1: float,
                            z2: float) -> ndarray:
        return CPUBackend._fast_3d_line(x, y, z, weight, h, weight_function,
                                        kernel_radius, pixels,
                                        x1, x2, y1, y2, z1, z2)

    @staticmethod
    def interpolate_3d_projection(x: ndarray,
                                  y: ndarray,
                                  weight: ndarray,
                                  h: ndarray,
                                  weight_function: CPUDispatcher,
                                  kernel_radius: float,
                                  x_pixels: int,
                                  y_pixels: int,
                                  x_min: float,
                                  x_max: float,
                                  y_min: float,
                                  y_max: float,
                                  exact: bool) -> ndarray:
        if exact:
            return CPUBackend._exact_3d_project(x, y, weight, h, x_pixels,
                                                y_pixels, x_min, x_max,
                                                y_min, y_max)
        return CPUBackend._fast_2d(x, y, np.zeros(x.size), 0, weight, h,
                                   weight_function, kernel_radius, x_pixels,
                                   y_pixels, x_min, x_max, y_min, y_max, 2)

    @staticmethod
    def interpolate_3d_projection_vec(x: ndarray,
                                      y: ndarray,
                                      weight_x: ndarray,
                                      weight_y: ndarray,
                                      h: ndarray,
                                      weight_function: CPUDispatcher,
                                      kernel_radius: float,
                                      x_pixels: int,
                                      y_pixels: int,
                                      x_min: float,
                                      x_max: float,
                                      y_min: float,
                                      y_max: float,
                                      exact: bool) -> Tuple[ndarray, ndarray]:
        if exact:
            return (CPUBackend._exact_3d_project(x, y, weight_x, h, x_pixels,
                                                 y_pixels, x_min, x_max,
                                                 y_min, y_max),
                    CPUBackend._exact_3d_project(x, y, weight_y, h, x_pixels,
                                                 y_pixels, x_min, x_max,
                                                 y_min, y_max))
        return (CPUBackend._fast_2d(x, y, np.zeros(x.size), 0, weight_x, h,
                                    weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 2),
                CPUBackend._fast_2d(x, y, np.zeros(y.size), 0, weight_y, h,
                                    weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 2))

    @staticmethod
    def interpolate_3d_cross(x: ndarray,
                             y: ndarray,
                             z: ndarray,
                             z_slice: float,
                             weight: ndarray,
                             h: ndarray,
                             weight_function: CPUDispatcher,
                             kernel_radius: float,
                             x_pixels: int,
                             y_pixels: int,
                             x_min: float,
                             x_max: float,
                             y_min: float,
                             y_max: float) -> ndarray:
        return CPUBackend._fast_2d(x, y, z, z_slice, weight, h,
                                   weight_function, kernel_radius, x_pixels,
                                   y_pixels, x_min, x_max, y_min, y_max, 3)

    @staticmethod
    def interpolate_3d_cross_vec(x: ndarray,
                                 y: ndarray,
                                 z: ndarray,
                                 z_slice: float,
                                 weight_x: ndarray,
                                 weight_y: ndarray,
                                 h: ndarray,
                                 weight_function: CPUDispatcher,
                                 kernel_radius: float,
                                 x_pixels: int,
                                 y_pixels: int,
                                 x_min: float,
                                 x_max: float,
                                 y_min: float,
                                 y_max: float) -> Tuple[ndarray, ndarray]:
        return (CPUBackend._fast_2d(x, y, z, z_slice, weight_x, h,
                                    weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 3),
                CPUBackend._fast_2d(x, y, z, z_slice, weight_y, h,
                                    weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 3))

    @staticmethod
    def interpolate_3d_grid(x: ndarray,
                            y: ndarray,
                            z: ndarray,
                            weight: ndarray,
                            h: ndarray,
                            weight_function: CPUDispatcher,
                            kernel_radius: float,
                            x_pixels: int,
                            y_pixels: int,
                            z_pixels: int,
                            x_min: float,
                            x_max: float,
                            y_min: float,
                            y_max: float,
                            z_min: float,
                            z_max: float) -> ndarray:
        image = np.zeros((z_pixels, y_pixels, x_pixels))
        pixwidthz = (z_max - z_min) / z_pixels

        for z_i in np.arange(z_pixels):
            z_val = z_min + (z_i + 0.5) * pixwidthz
            image[z_i] = CPUBackend._fast_2d(x, y, z, z_val, weight, h,
                                             weight_function, kernel_radius,
                                             x_pixels, y_pixels, x_min, x_max,
                                             y_min, y_max, 3)

        return image

    @staticmethod
    def interpolate_3d_mesh(x1: ndarray,
                            x2: ndarray,
                            x3: ndarray,
                            weight: ndarray,
                            h: ndarray,
                            weight_function: CPUDispatcher,
                            kernel_radius: float,
                            geometry: int,
                            x1_pixels: int,
                            x2_pixels: int,
                            x3_pixels: int,
                            x1_min: float,
                            x1_max: float,
                            x2_min: float,
                            x2_max: float,
                            x3_min: float,
                            x3_max: float) -> ndarray:
        image = np.zeros((x3_pixels, x2_pixels, x1_pixels))
        pixwidthx3 = (x3_max - x3_min) / x3_pixels

        if geometry == 0:
            for x3_i in np.arange(x3_pixels):
                x3_val = x3_min + (x3_i + 0.5) * pixwidthx3
                image[x3_i] = CPUBackend._fast_2d(
                                x1, x2, x3, x3_val, weight, h,
                                weight_function, kernel_radius,
                                x1_pixels, x2_pixels,
                                x1_min, x1_max, x2_min, x2_max, 3)
        elif geometry == 1:
            for x3_i in np.arange(x3_pixels):
                x3_val = x3_min + (x3_i + 0.5) * pixwidthx3
                image[x3_i] = CPUBackend._fast_2d_cyl(
                                x1, x2, x3, x3_val, weight, h,
                                weight_function, kernel_radius,
                                x1_pixels, x2_pixels,
                                x1_min, x1_max, x2_min, x2_max, 3)
        else:
            for x3_i in np.arange(x3_pixels):
                x3_val = x3_min + (x3_i + 0.5) * pixwidthx3
                image[x3_i] = CPUBackend._fast_2d_sph(
                                x1, x2, x3, x3_val, weight, h,
                                weight_function, kernel_radius,
                                x1_pixels, x2_pixels,
                                x1_min, x1_max, x2_min, x2_max, 3)

        return image

    # Underlying CPU numba-compiled code for interpolation to a 2D grid. Used
    # in interpolation of 2D data, and column integration / cross-sections of
    # 3D data.

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_2d_sph(x1_data: ndarray,
                     x2_data: ndarray,
                     x3_data: ndarray,
                     x3_slice: float,
                     w_data: ndarray,
                     h_data: ndarray,
                     weight_function: CPUDispatcher,
                     kernel_radius: float,
                     x1_pixels: int,
                     x2_pixels: int,
                     x1_min: float,
                     x1_max: float,
                     x2_min: float,
                     x2_max: float,
                     n_dims: int) -> ndarray:

        output = np.zeros((x2_pixels, x1_pixels))
        pixwidthx1 = (x1_max - x1_min) / x1_pixels
        pixwidthx2 = (x2_max - x2_min) / x2_pixels

        if not n_dims == 2:
            dx3 = np.subtract(np.float64(x3_slice), x3_data)
        else:
            dx3 = np.zeros(x1_data.size)

        term = w_data / h_data ** n_dims
        output_local = np.zeros((get_num_threads(), x2_pixels, x1_pixels))

        # thread safety:
        # each thread has its own grid, which are combined after interpolation
        for thread in prange(get_num_threads()):
            block_size = x1_data.size / get_num_threads()
            range_start = int(thread * block_size)
            range_end = int((thread + 1) * block_size)

            # iterate through the indexes of non-filtered particles
            for i in range(range_start, range_end):
                if np.abs(dx3[i]) >= kernel_radius * h_data[i]:
                    continue
                rad = kernel_radius * h_data[i]

                # Convert physical radius to angular extents
                drad1 = rad/(x3_data[i]*np.sin(x2_data[i]))
                drad2 = rad/x3_data[i]

                # determine pixels that this particle contributes to
                ipixmin = int(np.rint((x1_data[i]-drad1-x1_min)/pixwidthx1))
                jpixmin = int(np.rint((x2_data[i]-drad2-x2_min)/pixwidthx2))
                ipixmax = int(np.rint((x1_data[i]+drad1-x1_min)/pixwidthx1))
                jpixmax = int(np.rint((x2_data[i]+drad2-x2_min)/pixwidthx2))

                if ipixmax < 0 or ipixmin > x1_pixels:
                    continue
                if jpixmax < 0 or jpixmin > x2_pixels:
                    continue
                if ipixmin < 0:
                    ipixmin = 0
                if ipixmax > x1_pixels:
                    ipixmax = x1_pixels
                if jpixmin < 0:
                    jpixmin = 0
                if jpixmax > x2_pixels:
                    jpixmax = x2_pixels

                # precalculate differences (optimization)
                x1pix = x1_min+(np.arange(ipixmin, ipixmax)+0.5)*pixwidthx1
                x2pix = x2_min+(np.arange(jpixmin, jpixmax)+0.5)*pixwidthx2

                x3_2 = x3_slice*x3_slice + x3_data[i]*x3_data[i]
                x3_12 = x3_slice*x3_data[i]
                dx1 = (x1pix - x1_data[i]).reshape(1, x1pix.size)

                sinthe = np.sin(x2pix) * np.sin(x2_data[i])
                costhe = np.cos(x2pix) * np.cos(x2_data[i])

                sinthe = sinthe.reshape(x2pix.size, 1)
                costhe = costhe.reshape(x2pix.size, 1)

                r2 = x3_2 - 2.0 * x3_12 * (sinthe * np.cos(dx1) + costhe)

                h1i = 1.0/h_data[i]
                q = np.sqrt(r2)*h1i

                for jpix in range(jpixmax - jpixmin):
                    for ipix in range(ipixmax - ipixmin):
                        if q[jpix][ipix] > kernel_radius:
                            continue
                        wab = weight_function(q[jpix][ipix], n_dims)
                        jp = jpix + jpixmin
                        ip = ipix + ipixmin
                        output_local[thread][jp, ip] += term[i] * wab

        for i in range(get_num_threads()):
            output += output_local[i]

        return output

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_2d_cyl(x1_data: ndarray,
                     x2_data: ndarray,
                     x3_data: ndarray,
                     x3_slice: float,
                     w_data: ndarray,
                     h_data: ndarray,
                     weight_function: CPUDispatcher,
                     kernel_radius: float,
                     x1_pixels: int,
                     x2_pixels: int,
                     x1_min: float,
                     x1_max: float,
                     x2_min: float,
                     x2_max: float,
                     n_dims: int) -> ndarray:

        output = np.zeros((x2_pixels, x1_pixels))
        pixwidthx1 = (x1_max - x1_min) / x1_pixels
        pixwidthx2 = (x2_max - x2_min) / x2_pixels

        if not n_dims == 2:
            dx3 = np.subtract(np.float64(x3_slice), x3_data)
        else:
            dx3 = np.zeros(x1_data.size)

        term = w_data / h_data ** n_dims
        output_local = np.zeros((get_num_threads(), x2_pixels, x1_pixels))

        # thread safety:
        # each thread has its own grid, which are combined after interpolation
        for thread in prange(get_num_threads()):
            block_size = x1_data.size / get_num_threads()
            range_start = int(thread * block_size)
            range_end = int((thread + 1) * block_size)

            # iterate through the indexes of non-filtered particles
            for i in range(range_start, range_end):
                if np.abs(dx3[i]) >= kernel_radius * h_data[i]:
                    continue
                rad = kernel_radius * h_data[i]

                # Convert physical radius to angular extents
                drad1 = rad/x3_data[i]
                drad2 = rad

                # determine pixels that this particle contributes to
                ipixmin = int(np.rint((x1_data[i]-drad1-x1_min)/pixwidthx1))
                jpixmin = int(np.rint((x2_data[i]-drad2-x2_min)/pixwidthx2))
                ipixmax = int(np.rint((x1_data[i]+drad1-x1_min)/pixwidthx1))
                jpixmax = int(np.rint((x2_data[i]+drad2-x2_min)/pixwidthx2))

                if ipixmax < 0 or ipixmin > x1_pixels:
                    continue
                if jpixmax < 0 or jpixmin > x2_pixels:
                    continue
                if ipixmin < 0:
                    ipixmin = 0
                if ipixmax > x1_pixels:
                    ipixmax = x1_pixels
                if jpixmin < 0:
                    jpixmin = 0
                if jpixmax > x2_pixels:
                    jpixmax = x2_pixels

                # precalculate differences (optimization)
                x1pix = x1_min+(np.arange(ipixmin, ipixmax)+0.5)*pixwidthx1
                x2pix = x2_min+(np.arange(jpixmin, jpixmax)+0.5)*pixwidthx2

                x3_2 = x3_slice*x3_slice + x3_data[i]*x3_data[i]
                x3_12 = x3_slice*x3_data[i]
                dx1 = (x1pix - x1_data[i]).reshape(1, x1pix.size)
                dx2 = (x2_data[i] - x2pix).reshape(x2pix.size, 1)

                r2 = x3_2 - 2.0 * x3_12 * np.cos(dx1) + dx2*dx2

                h1i = 1.0/h_data[i]
                q = np.sqrt(r2)*h1i

                for jpix in range(jpixmax - jpixmin):
                    for ipix in range(ipixmax - ipixmin):
                        if q[jpix][ipix] > kernel_radius:
                            continue
                        wab = weight_function(q[jpix][ipix], n_dims)
                        jp = jpix + jpixmin
                        ip = ipix + ipixmin
                        output_local[thread][jp, ip] += term[i] * wab

        for i in range(get_num_threads()):
            output += output_local[i]

        return output

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_2d(x_data: ndarray,
                 y_data: ndarray,
                 z_data: ndarray,
                 z_slice: float,
                 w_data: ndarray,
                 h_data: ndarray,
                 weight_function: CPUDispatcher,
                 kernel_radius: float,
                 x_pixels: int,
                 y_pixels: int,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 n_dims: int) -> ndarray:
        output = np.zeros((y_pixels, x_pixels))
        pixwidthx = (x_max - x_min) / x_pixels
        pixwidthy = (y_max - y_min) / y_pixels
        if not n_dims == 2:
            dz = np.subtract(np.float64(z_slice), z_data)
        else:
            dz = np.zeros(x_data.size)

        term = w_data / h_data ** n_dims

        output_local = np.zeros((get_num_threads(), y_pixels, x_pixels))

        # thread safety:
        # each thread has its own grid, which are combined after interpolation
        for thread in prange(get_num_threads()):
            block_size = x_data.size / get_num_threads()
            range_start = int(thread * block_size)
            range_end = int((thread + 1) * block_size)

            # iterate through the indexes of non-filtered particles
            for i in range(range_start, range_end):
                if np.abs(dz[i]) >= kernel_radius * h_data[i]:
                    continue

                rad = kernel_radius * h_data[i]

                # determine pixels that this particle contributes to
                ipixmin = int(np.rint((x_data[i] - rad - x_min) / pixwidthx))
                jpixmin = int(np.rint((y_data[i] - rad - y_min) / pixwidthy))
                ipixmax = int(np.rint((x_data[i] + rad - x_min) / pixwidthx))
                jpixmax = int(np.rint((y_data[i] + rad - y_min) / pixwidthy))

                if ipixmax < 0 or ipixmin > x_pixels:
                    continue
                if jpixmax < 0 or jpixmin > y_pixels:
                    continue

                if ipixmin < 0:
                    ipixmin = 0
                if ipixmax > x_pixels:
                    ipixmax = x_pixels
                if jpixmin < 0:
                    jpixmin = 0
                if jpixmax > y_pixels:
                    jpixmax = y_pixels

                # precalculate differences in the x-direction (optimization)
                h1i = 1.0/h_data[i]
                dx2i = (x_min + (np.arange(ipixmin, ipixmax) + 0.5)
                        * pixwidthx - x_data[i])**2 + dz[i]**2

                # determine differences in the y-direction
                ypix = y_min + (np.arange(jpixmin, jpixmax) + 0.5) * pixwidthy
                dy = ypix - y_data[i]
                dy2 = dy * dy

                # calculate contributions at pixels i, j from particle at x, y
                q = np.sqrt(dx2i + dy2.reshape(len(dy2), 1)) * h1i

                for jpix in range(jpixmax - jpixmin):
                    for ipix in range(ipixmax - ipixmin):
                        if q[jpix][ipix] > kernel_radius:
                            continue
                        wab = weight_function(q[jpix][ipix], n_dims)
                        jp = jpix + jpixmin
                        ip = ipix + ipixmin
                        output_local[thread][jp, ip] += term[i] * wab

        for i in range(get_num_threads()):
            output += output_local[i]

        return output

    # Underlying CPU numba-compiled code for exact interpolation of 2D data to
    # a 2D grid.
    @staticmethod
    @njit(parallel=True)
    def _exact_2d_render(x_data: ndarray,
                         y_data: ndarray,
                         w_data: ndarray,
                         h_data: ndarray,
                         x_pixels: int,
                         y_pixels: int,
                         x_min: float,
                         x_max: float,
                         y_min: float,
                         y_max: float) -> ndarray:
        output_local = np.zeros((get_num_threads(), y_pixels, x_pixels))
        pixwidthx = (x_max - x_min) / x_pixels
        pixwidthy = (y_max - y_min) / y_pixels

        term = w_data / h_data ** 2

        for thread in prange(get_num_threads()):
            block_size = x_data.size / get_num_threads()
            range_start = int(thread * block_size)
            range_end = int((thread + 1) * block_size)

            # iterate through the indexes of non-filtered particles
            for i in range(range_start, range_end):

                rad = float(2 * h_data[i])

                # determine pixels that this particle contributes to
                ipixmin = int(np.floor((x_data[i] - rad - x_min) / pixwidthx))
                jpixmin = int(np.floor((y_data[i] - rad - y_min) / pixwidthy))
                ipixmax = int(np.ceil((x_data[i] + rad - x_min) / pixwidthx))
                jpixmax = int(np.ceil((y_data[i] + rad - y_min) / pixwidthy))

                if ipixmax < 0 or ipixmin >= x_pixels:
                    continue
                if jpixmax < 0 or jpixmin > y_pixels:
                    continue

                if ipixmin < 0:
                    ipixmin = 0
                if ipixmax > x_pixels:
                    ipixmax = x_pixels
                if jpixmin < 0:
                    jpixmin = 0
                if jpixmax > y_pixels:
                    jpixmax = y_pixels

                denom = 1 / np.abs(pixwidthx * pixwidthy) * h_data[i] ** 2

                # To calculate the exact surface integral of this pixel,
                # calculate the comprising line integrals at each boundary of
                # the square.
                if jpixmax >= jpixmin:
                    ypix = y_min + (jpixmin + 0.5) * pixwidthy
                    dy = ypix - y_data[i]

                    for ipix in range(ipixmin, ipixmax):
                        xpix = x_min + (ipix + 0.5) * pixwidthx
                        dx = xpix - x_data[i]

                        # Top Boundary
                        r0 = 0.5 * pixwidthy - dy
                        d1 = 0.5 * pixwidthx + dx
                        d2 = 0.5 * pixwidthx - dx
                        pixint = line_int(r0, d1, d2, h_data[i])
                        wab = pixint * denom

                        output_local[thread, jpixmin, ipix] += term[i] * wab

                if ipixmax >= ipixmin:
                    xpix = x_min + (ipixmin + 0.5) * pixwidthx
                    dx = xpix - x_data[i]

                    for jpix in range(jpixmin, jpixmax):
                        ypix = y_min + (jpix + 0.5) * pixwidthy
                        dy = ypix - y_data[i]

                        # Left Boundary
                        r0 = 0.5 * pixwidthx - dx
                        d1 = 0.5 * pixwidthy - dy
                        d2 = 0.5 * pixwidthy + dy
                        pixint = line_int(r0, d1, d2, h_data[i])
                        wab = pixint * denom

                        output_local[thread, jpix, ipixmin] += term[i] * wab

                for jpix in range(jpixmin, jpixmax):
                    ypix = y_min + (jpix + 0.5) * pixwidthy
                    dy = ypix - y_data[i]

                    for ipix in range(ipixmin, ipixmax):
                        xpix = x_min + (ipix + 0.5) * pixwidthx
                        dx = xpix - x_data[i]

                        # Bottom boundaries
                        r0 = 0.5 * pixwidthy + dy
                        d1 = 0.5 * pixwidthx - dx
                        d2 = 0.5 * pixwidthx + dx
                        pixint = line_int(r0, d1, d2, h_data[i])
                        wab = pixint * denom

                        output_local[thread, jpix, ipix] += term[i] * wab

                        # The negative value of the bottom boundary is equal to
                        # the value of the top boundary of the pixel below this
                        # pixel.
                        if jpix < jpixmax - 1:
                            output_local[thread, jpix+1, ipix] -= term[i] * wab

                        # Right Boundaries
                        r0 = 0.5 * pixwidthx + dx
                        d1 = 0.5 * pixwidthy + dy
                        d2 = 0.5 * pixwidthy - dy
                        pixint = line_int(r0, d1, d2, h_data[i])
                        wab = pixint * denom

                        output_local[thread, jpix, ipix] += term[i] * wab

                        # The negative value of the right boundary is equal to
                        # the value of the left boundary of the pixel to the
                        # right of this pixel.
                        if ipix < ipixmax - 1:
                            output_local[thread, jpix, ipix+1] -= term[i] * wab

        output = np.zeros((y_pixels, x_pixels))

        for i in range(get_num_threads()):
            output += output_local[i]

        return output

    # Underlying CPU numba-compiled code for 2D->1D cross-sections.
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_2d_line(x_data: ndarray,
                      y_data: ndarray,
                      w_data: ndarray,
                      h_data: ndarray,
                      weight_function: CPUDispatcher,
                      kernel_radius: float,
                      pixels: int,
                      x1: float,
                      x2: float,
                      y1: float,
                      y2: float) -> ndarray:
        # determine the slope of the cross-section line
        gradient = 0.0
        if not x2 - x1 == 0:
            gradient = (y2 - y1) / (x2 - x1)
        yint = y2 - gradient * x2

        # determine the fraction of the line that one pixel represents
        xlength = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        pixwidth = xlength / pixels
        xpixwidth = (x2 - x1) / pixels

        term = w_data / h_data ** 2

        # the intersections between the line and a particle's 'smoothing
        # circle' are found by solving a quadratic equation with the below
        # values of a, b, and c. if the determinant is negative, the particle
        # does not contribute to the cross-section, and can be removed.
        aa = 1 + gradient ** 2
        bb = 2 * gradient * (yint - y_data) - 2 * x_data
        det = x_data**2 + y_data**2 - 2 * yint * y_data \
            + yint**2 - (kernel_radius * h_data)**2
        det = bb ** 2 - 4 * aa * det

        # create a filter for particles that do not contribute
        filter = det >= 0
        det = np.sqrt(det)

        output = np.zeros(pixels)

        # the starting and ending x coordinates of the lines intersections with
        # a particle's smoothing circle
        rstart = ((-bb[filter] - det[filter])
                  / (2 * aa)).clip(a_min=x1, a_max=x2)
        rend = ((-bb[filter] + det[filter])
                / (2 * aa)).clip(a_min=x1, a_max=x2)

        # start and end distances that are within a particle's smoothing circle
        rstart = np.sqrt((rstart - x1)**2
                         + ((gradient * rstart + yint) - y1)**2)
        rend = np.sqrt((rend - x1)**2 + (((gradient * rend + yint) - y1)**2))

        # the maximum and minimum pixels that each particle contributes to.
        ipixmin = np.rint(rstart / pixwidth).clip(a_min=0, a_max=pixels)
        ipixmax = np.rint(rend / pixwidth).clip(a_min=0, a_max=pixels)

        output_local = np.zeros((get_num_threads(), pixels))

        # thread safety:
        # each thread has its own grid, which are combined after interpolation
        for thread in prange(get_num_threads()):

            block_size = len(x_data[filter]) / get_num_threads()
            range_start = thread * block_size
            range_end = (thread + 1) * block_size

            # iterate through the indices of all non-filtered particles
            for i in range(range_start, range_end):
                # determine contributions to all pixels for this particle
                xpix = x1 + (np.arange(int(ipixmin[i]), int(ipixmax[i]))
                             + 0.5) * xpixwidth
                ypix = gradient * xpix + yint
                dy = ypix - y_data[filter][i]
                dx = xpix - x_data[filter][i]

                q2 = (dx**2 + dy**2) / h_data[filter][i]**2
                wab = weight_function(np.sqrt(q2), 2)

                # add contributions to output total
                for ipix in range(int(ipixmax[i]) - int(ipixmin[i])):
                    ip = ipix + int(ipixmin[i])
                    output_local[thread][ip] += term[filter][i] * wab[ipix]

        for i in range(get_num_threads()):
            output += output_local[i]

        return output

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_3d_line(x_data: ndarray,
                      y_data: ndarray,
                      z_data: ndarray,
                      w_data: ndarray,
                      h_data: ndarray,
                      weight_function: CPUDispatcher,
                      kernel_radius: float,
                      pixels: int,
                      x1: float,
                      x2: float,
                      y1: float,
                      y2: float,
                      z1: float,
                      z2: float) -> ndarray:
        output_local = np.zeros((get_num_threads(), pixels))

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        ux, uy, uz = dx / length, dy / length, dz / length
        term = w_data / h_data ** 3

        for thread in prange(get_num_threads()):
            block_size = x_data.size / get_num_threads()
            range_start = int(thread * block_size)
            range_end = int((thread + 1) * block_size)

            for i in range(range_start, range_end):

                dx = x1 - x_data[i]
                dy = y1 - y_data[i]
                dz = z1 - z_data[i]
                delta = (ux * dx + uy * dy + uz * dz)**2 \
                    - (dx**2 + dy**2 + dz**2) + (kernel_radius * h_data[i])**2
                if delta < 0:
                    continue

                d1 = -(ux * dx + uy * dy + uz * dz) - np.sqrt(delta)
                d2 = -(ux * dx + uy * dy + uz * dz) + np.sqrt(delta)

                pixmin = min(max(0, round((d1 / length) * pixels)), pixels)
                pixmax = min(max(0, round((d2 / length) * pixels)), pixels)

                xpix = x1 + (np.arange(pixmin, pixmax)
                             + 0.5) * (x2 - x1) / pixels
                ypix = y1 + (np.arange(pixmin, pixmax)
                             + 0.5) * (y2 - y1) / pixels
                zpix = z1 + (np.arange(pixmin, pixmax)
                             + 0.5) * (z2 - z1) / pixels

                xdiff = xpix - x_data[i]
                ydiff = ypix - y_data[i]
                zdiff = zpix - z_data[i]

                q2 = (xdiff**2 + ydiff**2 + zdiff**2) / h_data[i]**2
                wab = weight_function(np.sqrt(q2), 3)

                for ipix in range(pixmax - pixmin):
                    output_local[thread][ipix + pixmin] += term[i] * wab[ipix]

        output = np.zeros(pixels)

        for i in range(get_num_threads()):
            output += output_local[i]

        return output

    @staticmethod
    @njit(parallel=True)
    def _exact_3d_project(x_data: ndarray,
                          y_data: ndarray,
                          w_data: ndarray,
                          h_data: ndarray,
                          x_pixels: int,
                          y_pixels: int,
                          x_min: float,
                          x_max: float,
                          y_min: float,
                          y_max: float) -> ndarray:
        output_local = np.zeros((get_num_threads(), y_pixels, x_pixels))
        pixwidthx = (x_max - x_min) / x_pixels
        pixwidthy = (y_max - y_min) / y_pixels

        norm3d = 1 / np.pi
        dfac = h_data ** 3 / (pixwidthx * pixwidthy * norm3d)
        term = norm3d * w_data / h_data ** 3

        for thread in prange(get_num_threads()):
            block_size = term.size / get_num_threads()
            range_start = int(thread * block_size)
            range_end = int((thread + 1) * block_size)

            # iterate through the indexes of non-filtered particles
            for i in range(range_start, range_end):

                rad = float(2 * h_data[i])

                # determine pixels that this particle contributes to
                ipixmin = int(np.floor((x_data[i] - rad - x_min) / pixwidthx))
                jpixmin = int(np.floor((y_data[i] - rad - y_min) / pixwidthy))
                ipixmax = int(np.ceil((x_data[i] + rad - x_min) / pixwidthx))
                jpixmax = int(np.ceil((y_data[i] + rad - y_min) / pixwidthy))

                # The width of the z contribution of this particle.
                # = 2 * kernel_radius * h[i], where kernel_radius is 2 for the
                # cubic spline kernel.
                pixwidthz = float(4 * h_data[i])

                if ipixmax < 0 or ipixmin >= x_pixels:
                    continue
                if jpixmax < 0 or jpixmin >= y_pixels:
                    continue

                if ipixmin < 0:
                    ipixmin = 0
                if ipixmax > x_pixels:
                    ipixmax = x_pixels
                if jpixmin < 0:
                    jpixmin = 0
                if jpixmax > y_pixels:
                    jpixmax = y_pixels

                for jpix in range(jpixmin, jpixmax):
                    ypix = y_min + (jpix + 0.5) * pixwidthy
                    dy = ypix - y_data[i]
                    for ipix in range(ipixmin, ipixmax):
                        xpix = x_min + (ipix + 0.5) * pixwidthx
                        dx = xpix - x_data[i]

                        q2 = (dx ** 2 + dy ** 2) / h_data[i] ** 2

                        if q2 < 4 + 3 * pixwidthx * pixwidthy / h_data[i] ** 2:
                            # Calculate the volume integral of this pixel by
                            # summing the comprising surface integrals of each
                            # surface of the cube.

                            # x-y surfaces
                            pixint = 2 * surface_int(0.5 * pixwidthz,
                                                     x_data[i], y_data[i],
                                                     xpix, ypix,
                                                     pixwidthx, pixwidthy,
                                                     h_data[i])

                            # x-z surfaces
                            pixint += surface_int(ypix - y_data[i]
                                                  + 0.5 * pixwidthy,
                                                  x_data[i], 0,
                                                  xpix, 0, pixwidthx,
                                                  pixwidthz, h_data[i])
                            pixint += surface_int(y_data[i] - ypix
                                                  + 0.5 * pixwidthy,
                                                  x_data[i], 0,
                                                  xpix, 0, pixwidthx,
                                                  pixwidthz, h_data[i])

                            # y-z surfaces
                            pixint += surface_int(xpix - x_data[i]
                                                  + 0.5 * pixwidthx,
                                                  0, y_data[i], 0,
                                                  ypix, pixwidthz,
                                                  pixwidthy, h_data[i])
                            pixint += surface_int(x_data[i] - xpix
                                                  + 0.5 * pixwidthx,
                                                  0, y_data[i], 0,
                                                  ypix, pixwidthz,
                                                  pixwidthy, h_data[i])

                            wab = pixint * dfac[i]

                            output_local[thread, jpix, ipix] += term[i] * wab

        output = np.zeros((y_pixels, x_pixels))

        for i in range(get_num_threads()):
            output += output_local[i]

        return output
