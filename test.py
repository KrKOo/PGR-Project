from abc import ABC
import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    x /= np.linalg.norm(x)
    return x

class Material:
    def __init__(self, color, reflection, diffuse_coefficient=1., specular_coefficient=1., specular_exponent=50):
        self.color = np.array(color)
        self.reflection = reflection
        self.diffuse_coefficient = diffuse_coefficient
        self.specular_coefficient = specular_coefficient
        self.specular_exponent = specular_exponent

class SceneObject(ABC):
    def __init__(self, position, meterial: Material):
        self.position = np.array(position)
        self.material = meterial
    
    def intersect(self, origin, direction):
        raise NotImplementedError()
    
    def get_normal(self, point):
        raise NotImplementedError()
     
class Sphere(SceneObject):
    def __init__(self, position, material: Material, radius):
        super().__init__(position, material)
        self.radius = np.array(radius)
        
    def intersect(self, origin, direction):
        a = np.dot(direction, direction)
        OS = origin - self.position
        b = 2 * np.dot(direction, OS)
        c = np.dot(OS, OS) - np.dot(self.radius, self.radius)
        disc = b * b - 4 * a * c
        if disc > 0:
            distSqrt = np.sqrt(disc)
            q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
            t0 = q / a
            t1 = c / q
            t0, t1 = min(t0, t1), max(t0, t1)
            if t1 >= 0:
                return t1 if t0 < 0 else t0
        return np.inf
    
    def get_normal(self, point):
        return normalize(point - self.position)
    
class Plane(SceneObject):
    def __init__(self, position, material, normal):
        super().__init__(position, material)
        self.normal = np.array(normal)
        
    def intersect(self, origin, direction):
        denom = np.dot(direction, self.normal)
        if np.abs(denom) < 1e-6:
            return np.inf
        d = np.dot(self.position - origin, self.normal) / denom
        if d < 0:
            return np.inf
        return d
    
    def get_normal(self, point):
        return self.normal
    
class Cube(SceneObject):
    def __init__(self, position, material: Material, size):
        super().__init__(position, material)
        self.size = np.array(size) / 2  # Half the size for easier calculations
        self.min_bounds = self.position - self.size
        self.max_bounds = self.position + self.size

    def intersect(self, origin, direction):
        t_min = (self.min_bounds - origin) / direction
        t_max = (self.max_bounds - origin) / direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return np.inf
        return t_near

    def get_normal(self, point):
        normal = np.zeros(3)
        for i in range(3):
            if abs(point[i] - self.min_bounds[i]) < 1e-6:
                normal[i] = -1
            elif abs(point[i] - self.max_bounds[i]) < 1e-6:
                normal[i] = 1
        return normalize(normal)

    
class Light:
    def __init__(self, position, color):
        self.position = np.array(position)
        self.color = np.array(color)

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_object(self, obj):
        self.objects.append(obj)
        
    def add_light(self, light):
        self.lights.append(light)

class Camera:
    def __init__(self, position, direction, fov, aspect_ratio, aperture, focal_length):
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture
        self.focal_length = focal_length
        self.distortion = 5000

    def apply_distortion(self, x, y):
        # Convert to normalized screen coordinates
        r = np.sqrt(x**2 + y**2)
        if r == 0:
            return x, y

        # Apply radial distortion
        k1, k2 = 0.8,0.05
        r_distorted = r * (1 + k1 * r**2 + k2 * r**4)
        
        # Scale back to distorted coordinates
        x_distorted = x * (r_distorted / r)
        y_distorted = y * (r_distorted / r)
        return x_distorted, y_distorted

    def get_ray(self, x, y, wavelength_shift=0):
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
class RayTracer:
    def __init__(self, width, height, scene):
        self.width = width
        self.height = height
        self.scene = scene
        self.specular_exponent = 50
        self.max_depth = 3
        self.samples_per_pixel = 3
        self.camera = Camera(
            position=np.array([0., 0.35, -1]),
            direction=np.array([0, -0.35, 1]),  # Smer pohľadu kamery
            fov=90,
            aspect_ratio=float(width) / height,
            aperture=0.2,
            focal_length=2,
        )
        self.screen_bounds = (-1., -1. / self.camera.aspect_ratio + .25, 1., 1. / self.camera.aspect_ratio + .25)
        
    def trace_ray(self, ray_origin, ray_direction):
        closest_intersection = np.inf
        for i, scene_object in enumerate(self.scene.objects):
            intersection_distance = scene_object.intersect(ray_origin, ray_direction)
            if intersection_distance < closest_intersection:
                closest_intersection, closest_object_index = intersection_distance, i
        if closest_intersection == np.inf:
            return
        closest_object = self.scene.objects[closest_object_index]
        intersection_point = ray_origin + ray_direction * closest_intersection
        normal_at_intersection = closest_object.get_normal(intersection_point)
        object_color = closest_object.material.color
        view_direction = normalize(self.camera.position - intersection_point)
        
        color_at_ray = np.zeros(3)

        for light in self.scene.lights:
            light_direction = normalize(light.position - intersection_point)

            shadow_intersections = []
            for idx, shadow_object in enumerate(self.scene.objects):
                if idx == closest_object_index:
                    continue
                shadow_intersection = shadow_object.intersect(intersection_point + normal_at_intersection * .0001, light_direction)
                shadow_intersections.append(shadow_intersection)

            if shadow_intersections and min(shadow_intersections) < np.inf:
                continue

            color_at_ray += closest_object.material.diffuse_coefficient * max(np.dot(normal_at_intersection, light_direction), 0) * object_color * light.color
            color_at_ray += closest_object.material.specular_coefficient * max(np.dot(normal_at_intersection, normalize(light_direction + view_direction)), 0) ** self.specular_exponent * light.color
        
        return closest_object, intersection_point, normal_at_intersection, color_at_ray
    
    def render(self):
        image = np.zeros((self.height, self.width, 3))
        wavelength_shifts = [0.002, 0.0, -0.002]
        center_x, center_y = self.width / 2, self.height / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)
        vignetting_strength = 0.8  # Controls the intensity of vignetting
        vignetting_exponent = 2   # Controls the sharpness of the effect

        for i, x in enumerate(np.linspace(self.screen_bounds[0], self.screen_bounds[2], self.width)):
            if i % 10 == 0:
                print(f"Rendering: {i / float(self.width) * 100:.2f}%")
            for j, y in enumerate(np.linspace(self.screen_bounds[1], self.screen_bounds[3], self.height)):

                col = np.zeros(3)
                for _ in range(self.samples_per_pixel):
                    # Trace separate rays for red, green, and blue
                    for k, shift in enumerate(wavelength_shifts):
                        ray_origin, ray_direction = self.camera.get_ray(x, y, wavelength_shift=shift)
                        depth = 0
                        reflection = 1.0
                        while depth < self.max_depth:
                            traced = self.trace_ray(ray_origin, ray_direction)
                            if not traced:
                                break
                            obj, M, N, col_ray = traced
                            ray_origin, ray_direction = M + N * 0.0001, normalize(ray_direction - 2 * np.dot(ray_direction, N) * N)
                            depth += 1
                            col[k] += reflection * col_ray[k]
                            reflection *= obj.material.reflection

                # Calculate vignetting factor
                pixel_radius = np.sqrt((i - center_x)**2 + (j - center_y)**2) / max_radius
                vignette_factor = 1 - vignetting_strength * (pixel_radius**vignetting_exponent)
                vignette_factor = max(0, vignette_factor)
                # Average color over samples
                image[self.height - j - 1, i, :] = np.clip(col * vignette_factor / self.samples_per_pixel, 0, 1)

        return image

scene = Scene()

scene.add_object(Sphere([.75, .1, 1.], Material([0., 0., 1.], .5), .5))
# scene.add_object(Sphere([-.75, .1, 2.25], Material([.5, .223, .5], .5), .5))
scene.add_object(Cube([-.75, .1, 2.25], Material([.5, .223, .5], .5), [1., 1., 1.]))
scene.add_object(Sphere([-2.75, .1, 3.5], Material([1., .572, .184], .5), .5))
# scene.add_object(Sphere([0, 0, 0], Material([1., 1., .184], .5), .5))
scene.add_object(Plane([0., -.5, 0.], Material([1., 1., 1.], .25), [0., 1., 0.]))

scene.add_light(Light([0., 5., -10.], np.ones(3)))
# scene.add_light(Light([10., 5., 10.],np.ones(3)))

rayTracer = RayTracer(1920, 1080, scene)

image = rayTracer.render()

plt.imsave('fig5.png', image)