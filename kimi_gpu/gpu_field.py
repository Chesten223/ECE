# gpu_field.py
import cupy as cp
import pygame
import numpy as np
import torch
from pygame.math import Vector2

WORLD_SIZE = 512  # 与主文件保持一致

class GPUField:
    def __init__(self, size, color, name):
        self.size = size
        self.color = color          # 0=红 1=绿 2=蓝
        self.name = name
        self.grid = cp.zeros((size, size), dtype=cp.float32)

    # 每帧衰减
    def update(self, dt):
        self.grid *= (1.0 - 0.001 * dt)

    # 圆形能量源
    def add_circular_source(self, pos, radius, value):
        x0, y0 = int(pos.x), int(pos.y)
        r = int(radius)
        y, x = cp.ogrid[max(0, y0-r):y0+r+1, max(0, x0-r):x0+r+1]
        mask = (x - x0)**2 + (y - y0)**2 <= r**2
        dist = cp.sqrt((x - x0)**2 + (y - y0)**2)
        new_vals = value * cp.maximum(0, 1 - (dist / radius)**0.6)
        self.grid[y, x] = cp.clip(self.grid[y, x] + new_vals * mask, 0, 1)

    # 采样值与梯度
    def get_value_and_gradient(self, pos):
        x = int(pos.x) % self.size
        y = int(pos.y) % self.size
        val = float(self.grid[y, x])
        gx = float(self.grid[y, (x+1) % self.size] - self.grid[y, (x-1) % self.size]) * 0.5
        gy = float(self.grid[(y+1) % self.size, x] - self.grid[(y-1) % self.size, x]) * 0.5
        return val, Vector2(gx, gy)

    # Pygame 渲染（CPU 回拷）
    def draw(self, surface, camera, alpha=128):
        import torch
        cpu_grid = torch.as_tensor(self.grid, device='cpu').numpy()
        # 颜色映射
        arr = np.zeros((*cpu_grid.shape, 3), dtype=np.uint8)
        arr[..., self.color] = (cpu_grid * 255).astype(np.uint8)
        img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        img.set_alpha(alpha)
        # 下面与 CPU 版缩放相同，可复用原 draw 代码
        # 为简单起见，直接缩放
        sw, sh = int(self.size * camera.zoom), int(self.size * camera.zoom)
        if sw > 0 and sh > 0:
            scaled = pygame.transform.scale(img, (sw, sh))
            sx, sy = camera.world_to_screen(Vector2(0, 0))
            surface.blit(scaled, (sx, sy))