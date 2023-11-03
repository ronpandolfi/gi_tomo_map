import numpy as np
import scipy
from PIL import Image
from alive_progress import alive_it, config_handler
from matplotlib import pyplot as plt

# This script uses a single 2d luminosity map (phantom_map.png) as a phantom domain map. The luminosity values
# correspond to different domain orientations, and should be understood to be cyclic around 0-255 (representing 0-180).

# The outputs of this script are matplotlib images showing:
# 1) Domain maps separated by domain angle
# 2) Sinograms generated with GI geometry via projection operator
# 3) Reconstructions of the original domain maps (1)

# NOTE: this synthetic data ignores that the bragg condition (and exclusion rules) would create phi gaps

config_handler.set_global(force_tty=True, max_cols=200, bar='filling')

if __name__ == '__main__':
    # read luminosity map
    fn = "phantom_map.png"
    image = Image.open(fn).convert('L')

    # convert to angle (180) map
    map = (np.asarray(image) / 255 * 180).astype(int) % 180

    map_size = map.shape[0]

    # print unique domain angles
    print('angles:', np.unique(map))

    # separate domain maps
    domain_maps = {angle: map == angle for angle in np.unique(map)}
    for angle, domain_map in domain_maps.items():
        plt.title(f'Domain Map @ Domain Angle: {angle} deg')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(domain_map, interpolation='nearest')
        plt.show()

    num = 2 * map_size  # number of steps along line cut
    slice_factor = 2  # factor controlling length of footprint; 1 = slice is same length as map width

    # compute sinograms pixelwise
    # domain_sinograms = {}
    # for domain_angle, domain_map in alive_it(domain_maps.items(), title='Calculating sinograms pixelwise'):
    #     domain_sinogram = np.zeros((180, map_size))
    #     for x in range(0, map_size):
    #         for phi in range(0, 180):
    #             # get a sampling over a slice through x, phi
    #             x0 = x + slice_factor * map_size / 2 * np.sin(np.deg2rad(phi))
    #             y0 = map_size / 2 + slice_factor * map_size / 2 * np.cos(np.deg2rad(phi))
    #             x1 = x - slice_factor * map_size / 2 * np.sin(np.deg2rad(phi))
    #             y1 = map_size / 2 - slice_factor * map_size / 2 * np.cos(np.deg2rad(phi))
    #             xs, ys = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    #
    #             # show slice overlay
    #             # plt.imshow(map)
    #             # plt.plot([x0, x1], [y0, y1], 'ro-')
    #             # plt.show()
    #             # input()
    #
    #             # (only sample the current angle)
    #             slice = scipy.ndimage.map_coordinates(domain_map, np.vstack((ys, xs)),
    #                                                   order=0,
    #                                                   mode='constant',
    #                                                   cval=0)
    #
    #             # sum up slice to represent signal amplitude for current domain angle through slice
    #             I = np.sum(slice)
    #
    #             domain_sinogram[phi, x] = I
    #     domain_sinograms[domain_angle] = domain_sinogram

    projection_operator = np.empty((map_size, 180, map_size, map_size))
    for x in alive_it(range(map_size), title='Constructing projection operator', total=map_size):
        for phi in range(180):
            # get a sampling over a slice through x, phi
            x0 = x + slice_factor * map_size / 2 * np.sin(np.deg2rad(phi))
            y0 = map_size / 2 + slice_factor * map_size / 2 * np.cos(np.deg2rad(phi))
            x1 = x - slice_factor * map_size / 2 * np.sin(np.deg2rad(phi))
            y1 = map_size / 2 - slice_factor * map_size / 2 * np.cos(np.deg2rad(phi))
            xs, ys = np.around(np.linspace(x0, x1, num)).astype(int), np.around(np.linspace(y0, y1, num)).astype(int)

            # show slice overlay
            # plt.imshow(map)
            # plt.plot([x0, x1], [y0, y1], 'ro-')
            # plt.show()
            # input()

            # mask positions out of bounds
            xy_mask = (xs >= 0) & (xs < map_size) & (ys >= 0) & (ys < map_size)
            masked_xs, masked_ys = xs[xy_mask], ys[xy_mask]

            # construct a mask in the map space
            map_mask = np.zeros_like(map)
            coords = np.vstack([masked_ys, masked_xs])
            np.put(map_mask, np.ravel_multi_index(coords, map_mask.shape), 1)

            # stash the mask into the projection operator
            projection_operator[x, phi] = map_mask

    # reshape and sparsify the projection operator
    projection_operator = scipy.sparse.coo_array(projection_operator.reshape(map_size * 180, -1))

    # compute sinograms by application of projection operator
    domain_sinograms_projection_operator = {}
    for domain_angle, domain_map in alive_it(domain_maps.items(), title='Computing sinograms via projection operator'):
        domain_sinograms_projection_operator[domain_angle] = (projection_operator @ domain_map.ravel()).reshape(map_size, 180).T

    # Compare sinograms computed pixelwise vs by projection operator
    # for (angle, sino), operated_sino in zip(domain_sinograms.items(), domain_sinograms_projection_operator.values()):
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.set_title(f'Pixelwise\nIntegrated Sinogram\n@ Domain Angle: {angle} deg')
    #     ax1.set(xlabel='x', ylabel='ϕ')
    #     ax1.imshow(sino, vmin=0, vmax=sino.max())
    #     ax2.set_title(f'Projection Operator\nIntegrated Sinogram \n@ Domain Angle: {angle} deg')
    #     ax2.set(xlabel='x', ylabel='ϕ')
    #     ax2.imshow(operated_sino, vmax=operated_sino.max(), vmin=0)
    #     plt.show()

    # Reconstruct the original domain orientation maps
    recon = {}
    for angle, sino in alive_it(domain_sinograms_projection_operator.items(), title='Calculating reconstructions'):
        # recon[angle] = sirt(sino, projection_operator)
        recon[angle] = scipy.sparse.linalg.lsqr(projection_operator, sino.ravel())[0].reshape(map_size, map_size)

    # Show reconstructions
    for angle, domain_recon in recon.items():
        plt.title(f'Recon Map @ Domain Angle: {angle} deg')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(domain_recon, interpolation='nearest')
        plt.show()
