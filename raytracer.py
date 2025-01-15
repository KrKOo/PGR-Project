import numpy as np
from camera import Camera
from scene import Scene
from utils import normalize

class RayTracer:
    def __init__(
            self, 
            width: int, 
            height: int, 
            scene: Scene, 
            camera: Camera, 
            max_depth: int = 3, 
            samples_per_pixel: int = 1):
        self.width = width
        self.height = height
        self.scene = scene
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel
        self.camera = camera

        self.screen_bounds = (-1., -1. / self.camera.aspect_ratio + .25, 1., 1. / self.camera.aspect_ratio + .25)
        
    def trace_ray(self, ray_origin: np.array, ray_direction: np.array):
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
            color_at_ray += closest_object.material.specular_coefficient * max(np.dot(normal_at_intersection, normalize(light_direction + view_direction)), 0) ** closest_object.material.specular_exponent * light.color
        
        return closest_object, intersection_point, normal_at_intersection, color_at_ray
    
    def render(self):
        image = np.zeros((self.height, self.width, 3))
        center_x, center_y = self.width / 2, self.height / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)

        for i, x in enumerate(np.linspace(self.screen_bounds[0], self.screen_bounds[2], self.width)):
            if i % 10 == 0:
                print(f"Rendering: {i / float(self.width) * 100:.2f}%")
            for j, y in enumerate(np.linspace(self.screen_bounds[1], self.screen_bounds[3], self.height)):

                col = np.zeros(3)
                for _ in range(self.samples_per_pixel):
                    # Trace separate rays for red, green, and blue
                    for k, shift in enumerate(self.camera.wavelength_shifts):
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
                vignette_factor = 1 - self.camera.vignetting_strength * (pixel_radius**self.camera.vignetting_exponent)
                vignette_factor = max(0, vignette_factor)
                # Average color over samples
                image[self.height - j - 1, i, :] = np.clip(col * vignette_factor / self.samples_per_pixel, 0, 1)

        return image