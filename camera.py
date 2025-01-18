
import numpy as np
from utils import normalize


class Camera:
    def __init__(
            self, 
        	position: np.array,
            direction: np.array,
            aspect_ratio: float,
            aperture: float,
            focal_length: float,
            distortion_coefficients: np.array,
            wavelength_shifts: np.array,
            vignetting_strength: float,
            vignetting_exponent: float):
        self.position = position
        self.direction = direction
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture
        self.focal_length = focal_length
        self.distortion_coefficients = distortion_coefficients
        self.wavelength_shifts = wavelength_shifts
        self.vignetting_strength = vignetting_strength
        self.vignetting_exponent = vignetting_exponent

    def apply_distortion(self, x: float, y: float):
        # Convert to normalized screen coordinates
        r = np.sqrt(x**2 + y**2)

        # Apply radial distortion
        k1, k2 = self.distortion_coefficients

        r_distorted = r * (1 + k1 * r**2 + k2 * r**4)
        
        # Scale back to distorted coordinates
        x_distorted = x * (r_distorted / r)
        y_distorted = y * (r_distorted / r)
        return x_distorted, y_distorted

    def get_ray(self, x: float, y: float, wavelength_shift: float = 0):
        x, y = self.apply_distortion(x, y)

        radius = self.aperture / 2
        random_point = radius * np.random.rand(2) * np.array([
            np.cos(2 * np.pi * np.random.rand()), 
            np.sin(2 * np.pi * np.random.rand())
        ])
        offset = np.array([random_point[0], random_point[1], 0])

        # Adjust direction for chromatic aberration
        adjusted_direction = normalize(np.array([x + wavelength_shift, y + wavelength_shift, 0]) - self.position)

        # Depth of field
        focus_point = self.position + self.focal_length * adjusted_direction
        adjusted_direction = normalize(focus_point - (self.position + offset))

        return (self.position + offset, adjusted_direction)