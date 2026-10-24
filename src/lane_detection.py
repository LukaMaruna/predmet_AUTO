"""Lane line pixels in the binary top-down image: column histogram, sliding windows, polynomial fit."""
import numpy as np

# Sliding window search
NWINDOWS = 9      # windows stacked from the bottom to the top of the image
MARGIN = 100      # half width of a window, px
MINPIX = 50       # pixels needed in a window to re-centre the next one


def column_histogram(binary):
    """Number of line pixels in every column of the lower half of the image."""
    height = binary.shape[0]
    return np.sum(binary[height // 2:, :] > 0, axis=0)


def line_bases(histogram):
    """Columns of the highest histogram peak in the left and in the right half."""
    middle = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:middle])
    right_base = np.argmax(histogram[middle:]) + middle
    return left_base, right_base


def sliding_window(binary, x_base):
    """Follow one line upwards from x_base; returns its pixel coordinates and the windows."""
    height = binary.shape[0]
    window_height = height // NWINDOWS
    nonzeroy, nonzerox = binary.nonzero()

    x_current = x_base
    line_indices, windows = [], []
    for window in range(NWINDOWS):
        y_low = height - (window + 1) * window_height
        y_high = height - window * window_height
        x_low = x_current - MARGIN
        x_high = x_current + MARGIN
        windows.append((x_low, y_low, x_high, y_high))

        # Line pixels inside this window
        inside = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                  (nonzerox >= x_low) & (nonzerox < x_high)).nonzero()[0]
        line_indices.append(inside)

        # Centre the next window on these pixels if there are enough of them
        if len(inside) > MINPIX:
            x_current = int(np.mean(nonzerox[inside]))

    line_indices = np.concatenate(line_indices)
    return nonzeroy[line_indices], nonzerox[line_indices], windows


def fit_poly(ys, xs):
    """Second order polynomial x = a*y^2 + b*y + c through the line pixels."""
    return np.polyfit(ys, xs, 2)


def find_lane_lines(binary):
    """Detect both lane lines; returns a dict per side with pixels, windows and fit."""
    histogram = column_histogram(binary)
    bases = line_bases(histogram)
    lines = {}
    for side, x_base in zip(("left", "right"), bases):
        ys, xs, windows = sliding_window(binary, x_base)
        lines[side] = {"base": x_base, "ys": ys, "xs": xs, "windows": windows, "fit": fit_poly(ys, xs)}
    return lines, histogram
