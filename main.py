import numpy as np
import matplotlib.pyplot as plt

from scene import Scene, Material, Sphere, Plane, Cube, Light
from camera import Camera
from raytracer import RayTracer

WIDTH, HEIGHT = 200, 200

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

scene.add_light(
	Light(
		position=[0, 5, -10], 
		color=[1, 1, 1])
	)

camera = Camera(
	position=np.array([0, 0.35, -1]),
	direction=np.array([0, -0.35, 1]),  # Smer pohľadu kamery
	aspect_ratio=WIDTH/HEIGHT,
	aperture=0.2,
	focal_length=2,
	distortion_coefficients=np.array([-0.3, 0.15]),
	wavelength_shifts=np.array([0.002, 0, 0.002]),
	vignetting_strength=0.5,
	vignetting_exponent=1.5
)

rayTracer = RayTracer(WIDTH, HEIGHT, scene, camera)

image = rayTracer.render()

plt.imsave('fig7.png', image)