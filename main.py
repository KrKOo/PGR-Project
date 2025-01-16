import numpy as np
import matplotlib.pyplot as plt

from scene import Scene, Material, Sphere, Plane, Cube, Light
from camera import Camera
from raytracer import RayTracer

# Resolution of the final rendered image
WIDTH, HEIGHT = 1920, 1080

# Create a new scene
scene = Scene()

scene.add_object(
    Sphere(
        position=[0.75, 0.1, 1], 
        material=Material([0, 0, 1], 0.5),
        radius=0.5)
    )

scene.add_object(
    Cube(
        position=[-0.75, 0.1, 2.25],
        material=Material([0.5, 0.2, 0.5], 0.5),
        size=[1, 1, 1])
    )
scene.add_object(
    Sphere(
        position=[-2.75, 0.1, 3.5], 
        material=Material([1, 0.5, 0.2], 0.5), 
        radius=0.5)
    )

scene.add_object(
    Plane(
        position=[0, -0.5, 0], 
        material=Material([1, 1, 1], 0.25),
        normal=[0, 1, 0])
    )

# Add a light source above and behind the camera
scene.add_light(
    Light(
        position=[0, 5, -10], 
        color=[1, 1, 1])
    )

# Create a camera with some lens/distortion settings
camera = Camera(
    position=np.array([0, 0.35, -1]),       		# Camera location
    direction=np.array([0, -0.35, 1]),      		# Camera viewing direction
    aspect_ratio=WIDTH/HEIGHT,              		# Match to image ratio
    aperture=0.2,                           		# For depth of field
    focal_length=2,                         		# Scene focus distance
    distortion_coefficients=np.array([0.2, 0.05]), 	# Lens distortion
    wavelength_shifts=np.array([0.002, 0, -0.002]),	# Slight color dispersion
    vignetting_strength=0.5,                		# Darkening at edges
    vignetting_exponent=1.5                 		# Controls how quickly edges darken
)

# Initialize the raytracer
rayTracer = RayTracer(WIDTH, HEIGHT, scene, camera, max_depth=3, samples_per_pixel=10)

# Render the scene into a 2D image array
image = rayTracer.render()

# Save the final image
plt.imsave('data/fig10.png', image)