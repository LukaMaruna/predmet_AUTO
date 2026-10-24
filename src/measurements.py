"""Radius of curvature and vehicle offset from the lane centre, in metres."""
import numpy as np


def line_x(fit, y):
    """x of the polynomial x = a*y^2 + b*y + c at row y."""
    a, b, c = fit
    return a * y ** 2 + b * y + c


def radius_of_curvature(ys, xs, xm_per_pix, ym_per_pix, y_eval):
    """Radius of a line at row y_eval, from a polynomial refitted in metres."""
    A, B, C = np.polyfit(ys * ym_per_pix, xs * xm_per_pix, 2)
    y = y_eval * ym_per_pix
    return (1 + (2 * A * y + B) ** 2) ** 1.5 / abs(2 * A)


def vehicle_offset(left_fit, right_fit, y_eval, vehicle_x, xm_per_pix):
    """Distance of the vehicle from the lane centre; positive means right of the centre."""
    lane_centre = (line_x(left_fit, y_eval) + line_x(right_fit, y_eval)) / 2
    return (vehicle_x - lane_centre) * xm_per_pix


def measure(lines, scale, height):
    """Radius of both lines, lane radius (their mean) and vehicle offset at the image bottom."""
    y_eval = height - 1
    left, right = lines["left"], lines["right"]
    left_radius = radius_of_curvature(left["ys"], left["xs"], scale["xm_per_pix"], scale["ym_per_pix"], y_eval)
    right_radius = radius_of_curvature(right["ys"], right["xs"], scale["xm_per_pix"], scale["ym_per_pix"], y_eval)
    offset = vehicle_offset(left["fit"], right["fit"], y_eval, scale["vehicle_x"], scale["xm_per_pix"])
    return {"left_radius": left_radius, "right_radius": right_radius,
            "radius": (left_radius + right_radius) / 2, "offset": offset}
