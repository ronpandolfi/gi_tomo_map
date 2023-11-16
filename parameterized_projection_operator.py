import numpy as np
from matplotlib import pyplot as plt


def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def projection_operator(x, phi, map_size, center=None, width=1, length=None):
    if not length:
        length = map_size * 2

    if not center:
        center = (map_size/2, map_size/2)

    # get a sampling over a slice through x, phi
    v = np.array([-np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))]) * (-x + map_size/2)
    c = np.array(center)
    f = np.array([-np.sin(np.deg2rad(phi)), -np.cos(np.deg2rad(phi))]) * length / 2

    f0 = v + c + f
    f1 = v + c - f

    x0, y0 = f0
    x1, y1 = f1

    # draw a 1px line (with AA) as a mask from f0 to f1
    # NOTE: this is correct when the beam width is the same as the x/y resolution; for other cases,
    # you could use scipy.ndimage.zoom()
    rows, cols, vals = weighted_line(y0, x0, y1, x1, w=0.25)
    xy_mask = (cols >= 0) & (cols < map_size) & (rows >= 0) & (rows < map_size)
    masked_cols, masked_rows, masked_vals = cols[xy_mask], rows[xy_mask], vals[xy_mask]
    (map_mask := np.zeros((map_size, map_size)))[masked_rows, masked_cols] = masked_vals

    return map_mask


if __name__ == '__main__':

    for x in range(50, 256):
        for phi in range(0, 180):
            p = projection_operator(x, phi, 256)

            # show slice overlay
            plt.cla()
            plt.imshow(p, interpolation='nearest')
            plt.pause(0.001)
