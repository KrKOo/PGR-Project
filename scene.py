from abc import ABC
import numpy as np
from utils import normalize

class Material:
    def __init__(
            self, 
            color: np.array, 
            reflection: float, 
            diffuse_coefficient: float=1., 
            specular_coefficient: float=1.,
            specular_exponent: float=50):
        self.color = np.array(color)
        self.reflection = reflection
        self.diffuse_coefficient = diffuse_coefficient
        self.specular_coefficient = specular_coefficient
        self.specular_exponent = specular_exponent

class SceneObject(ABC):
    def __init__(self, position: np.array, meterial: Material):
        self.position = np.array(position)
        self.material = meterial
    
    def intersect(self, origin: np.array, direction: np.array):
        raise NotImplementedError()
    
    def get_normal(self, point: np.array):
        raise NotImplementedError()
     
class Sphere(SceneObject):
    def __init__(self, position: np.array, material: Material, radius):
        super().__init__(position, material)
        self.radius = np.array(radius)
        
    def intersect(self, origin: np.array, direction: np.array):
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
    
    def get_normal(self, point: np.array):
        return normalize(point - self.position)
    
class Plane(SceneObject):
    def __init__(self, position: np.array, material: Material, normal: np.array):
        super().__init__(position, material)
        self.normal = np.array(normal)
        
    def intersect(self, origin: np.array, direction: np.array):
        denom = np.dot(direction, self.normal)
        if np.abs(denom) < 1e-6:
            return np.inf
        d = np.dot(self.position - origin, self.normal) / denom
        if d < 0:
            return np.inf
        return d
    
    def get_normal(self, point: np.array):
        return self.normal
    
class Cube(SceneObject):
    def __init__(self, position: np.array, material: Material, size: np.array):
        super().__init__(position, material)
        self.size = np.array(size) / 2  # Half the size for easier calculations
        self.min_bounds = self.position - self.size
        self.max_bounds = self.position + self.size

    def intersect(self, origin: np.array, direction: np.array):
        t_min = (self.min_bounds - origin) / direction
        t_max = (self.max_bounds - origin) / direction

        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return np.inf
        return t_near

    def get_normal(self, point: np.array):
        normal = np.zeros(3)
        for i in range(3):
            if abs(point[i] - self.min_bounds[i]) < 1e-6:
                normal[i] = -1
            elif abs(point[i] - self.max_bounds[i]) < 1e-6:
                normal[i] = 1
        return normalize(normal)

    
class Light:
    def __init__(self, position: np.array, color: np.array):
        self.position = np.array(position)
        self.color = np.array(color)

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_object(self, obj: SceneObject):
        self.objects.append(obj)
        
    def add_light(self, light: Light):
        self.lights.append(light)