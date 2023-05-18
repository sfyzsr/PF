from dataclasses import dataclass, field
from random import gauss, uniform
import numpy as np
import math
import copy

@dataclass
class Particle:
    x: float
    y: float
    a: float
    scale: float = 1.0
    w: float = 1.0
    history: list = field(init=False, repr=False)

    def __post_init__(self):
        self.history = []
    
    def move(self, x: float, y: float, a: float):
        cos = math.cos(self.a)
        sin = math.sin(self.a)
        dx = x*cos - y*sin
        dy = x*sin + y*cos
        self.x += self.scale*dx
        self.y += self.scale*dy
        self.a += a

    def set_pose(self, x, y, a):
        self.x, self.y, self.a = x, y, a

    def record_history(self):
        self.history.append([self.x, self.y])

    def copy(self):
        return copy.deepcopy(self)

@dataclass
class Cloud:
    n: int
    initial_particle: Particle
    particles: list = field(init=False, repr=False)

    def __post_init__(self):
        self.particles = [self.initial_particle.copy() for i in range(self.n)]

    def estimate(self):
        pos = []
        for p in self.particles:
            pos.append([p.x,p.y])
        pos = np.array(pos)
        mean = np.average(pos, weights=np.array(self.weights()), axis=0)
        return mean
    
    def estimate_xya(self):
        pos = []
        for p in self.particles:
            pos.append([p.x, p.y, p.a])
        pos = np.array(pos)
        mean = np.average(pos, weights=np.array(self.weights()), axis=0)
        return mean, pos

    def move(self, control, devs = (0, 0, 0)):
        x, y, a = control[:3]
        sx, sy, sa = devs[:3]
        for p in self.particles:
            da = a*(1+gauss(0, sa))
            p.move(x*(1+gauss(0, sx)), y*(1+gauss(0, sy)), da)

    def set_pose(self, x, y, a):
        for p in self.particles:
            p.set_pose(x, y, a)
    
    def roughen(self, devs):
        for p in self.particles:
            sx, sy, sa = devs[:3]
            p.move(gauss(0, sx), gauss(0, sy), gauss(0, sa))


    def record_history(self):
        for p in self.particles:
            p.record_history()

    def mean_history(self):
        history = []
        
        for t in range(len(self.particles[0].history)):
            sx = 0.0
            sy = 0.0
            for p in self.particles:
                x, y = p.history[t]
                sx += x
                sy += y
            history.append((sx/self.n, sy/self.n))

        return history

    def update_weights(self, likelihood):
        for p in self.particles:
            p.w *= likelihood(p)

    def update_weights_dpf(self, likelihood):
        i = 0
        for p in self.particles:
            p.w *= likelihood[i]
            i += 1

    def weights(self):
        return [p.w for p in self.particles]

    def normalize_weights(self):
        s = sum(self.weights())
        if s > 0:
            for p in self.particles:
                p.w /= s
        else:
            self.reset_weights()
    
    def reset_weights(self):
        for p in self.particles:
            p.w = 1.0/self.n

    def neff(self):
        self.normalize_weights()
        s_sq = sum([w*w for w in self.weights()])
        return 1.0/s_sq

    def neff_ratio(self):
        return self.neff()/self.n

    # standard systematic resampling
    def resample(self):
        self.normalize_weights()
        cs = 0
        css = []
        for p in self.particles:
            cs += p.w
            css.append(cs)

        new_particles = []
        step = 1/self.n
        sum_iter = uniform(0, step)
        p_iter = 0

        while sum_iter < 1.0:
            if css[p_iter] >= sum_iter:
                new_particles.append(self.particles[p_iter].copy())
                sum_iter += step
            else:
                p_iter += 1

        self.particles = new_particles
        self.reset_weights()

