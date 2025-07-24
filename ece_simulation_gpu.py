# -*- coding: utf-8 -*-
# ====================================================
# 涌现认知生态系统（ECE） GPU加速版 v0.1
#
# 教学版本，Gemini（哈基米）和 七叶怀瑾 共同编写
# 日期： 2025/7/19
#
# v0.1 核心目标：搭建程序基本框架
# ====================================================

import pygame
import numpy as np
import random

import cupy as cp
from numba import cuda
import os

# --- 第一部分 宇宙公理 ---
WORLD_SIZE = 512
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800


# 1.5部分 定义核心类


class Field:
    """
    Field class representing a 2D grid of information in the universe
    Field 类，宇宙的二维网格类，用于存储和管理宇宙中的信息。
    """
    def __init__(self, size, name):
        """
        Constructor for the Field.
        Field 的构造函数。

        Args:
            size (int) : The size of one side of the square grid. 方形网格的边长。
            name (str) : The name of the field, for identification. 场的名字，用于识别。
        """
        # Store the name for later use.
        self.name = name

        # Create a square grid of the given size, initialized to all zeros.
        # We use np.float32 for performance, as it ueses less memory than the default float64.
        # 创建一个给定大小的方形网络，所有值初始化为0.
        # 我们使用np.float32是为了性能，因为它比默认的float64占用更少内存。
        self.grid = np.zeros((size, size), dtype=np.float32)
    def add_circular_source(self, pos, radius, value):
        """
        [CORRECTED VERSION] Adds a circular source with proper blending.
        [修正版] 添加一个能正确融合的圆形源。
        """
        # Get integer coordinates and radius.
        # 获取整数坐标和半径。
        x_center, y_center = int(pos[0]), int(pos[1])
        rad = int(radius)

        # --- New, Simplified, and Correct Logic ---
        # --- 全新的、简化的、正确的逻辑 ---

        # 1. Define the bounding box of the circle on the main grid.
        #    在主网格上定义这个圆的边界框。
        x_min = max(0, x_center - rad)
        x_max = min(self.grid.shape[1], x_center + rad)
        y_min = max(0, y_center - rad)
        y_max = min(self.grid.shape[0], y_center + rad)

        # If the bounding box is invalid, do nothing.
        # 如果边界框无效，则不执行任何操作。
        if x_min >= x_max or y_min >= y_max:
            return

        # 2. Create a coordinate grid FOR THE BOUNDING BOX ITSELF.
        #    直接为这个边界框创建坐标网格。
        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        # 3. Calculate the squared distance of each point in the box from the circle's true center.
        #    计算框内每个点到圆形真正中心的距离的平方。
        dist_sq = (x_coords - x_center)**2 + (y_coords - y_center)**2

        # 4. Create a mask for points that are truly inside the circle.
        #    为真正处于圆形内的点创建一个遮罩。
        circle_mask = dist_sq <= rad**2
        
        # 5. Calculate the gradient values ONLY for the points inside the circle.
        #    只为圆内的点计算梯度值。
        # First, get the distances (we take the sqrt only on the needed values for efficiency).
        # 首先获取距离（为了效率，我们只对需要的值开方）。
        distances = np.sqrt(dist_sq[circle_mask])
        
        # Then, calculate the gradient.
        # 然后，计算梯度。
        gradient_values_to_add = value * (1 - distances / rad)

        # 6. Apply the blending logic.
        #    应用融合逻辑。
        # Get the current values from our grid slice where the circle is.
        # 从我们的网格切片中，获取圆形所在位置的当前值。
        current_values = self.grid[y_min:y_max, x_min:x_max][circle_mask]

        # Use the blending formula: new = a + b - a*b
        # 使用融合公式: 新值 = A + B - A*B
        blended_values = current_values + gradient_values_to_add - (current_values * gradient_values_to_add)

        # 7. Assign the new blended values back to the grid.
        #    将融合后的新值赋回给网格。
        self.grid[y_min:y_max, x_min:x_max][circle_mask] = blended_values




    def draw(self, surface):
        # This method remains the same as the last version.
        # 这个方法和上一版保持一致。
        color_array = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        color_array[:, :, 1] = (np.clip(self.grid, 0, 1) * 255).astype(np.uint8)
        field_surface = pygame.surfarray.make_surface(color_array)
        field_surface.set_alpha(128)
        temp_surface = pygame.transform.rotate(field_surface, 90)
        flipped_surface = pygame.transform.flip(temp_surface, False, True)
        pygame.transform.scale(flipped_surface, surface.get_size(), surface)


class Universe:
    """
    Universe class, the simulator and manager of our entire simulated world.
    It contains everything and drives everything.
    宇宙类，我们整个模拟世界的模拟器和管理者。
    它包含一切，并驱动一切。
    """
    def __init__(self):
        """
        Constructor of the universe. This function is called when a universe is created.
        The self keyword represents the universe instance itself, allowing it to access its own variables and methods.
        宇宙的构造函数。当一个宇宙被创造的时候，这个函数会被调用。
        self关键字代表宇宙实例本身，让他能访问自己的变量和方法。
        """
        self.world_size = WORLD_SIZE
        self.agents = []  # Initialize an empty list to store all agents 初始化一个空的列表，存储所有智能体
        self.frame_count = 0 # Universe time starts from zero 宇宙的时间从零开始

        # --- Fields ---
        # Create an instance of our new field class for nutrients
        self.nutrient_field = Field(self.world_size, "Nutrient Field")

        # --- Initial Seeding ---
        # Call the method to seek initial energy patches
        self._initial_energy_seeding()

    def _initial_energy_seeding(self):
        """
        A private helper method to create initial patches of energy
        The underscore at the start suggests this is for internal use.
        """
        num_patches = 5
        for _ in range(num_patches):
            pos = (random.randint(0, self.world_size), random.randint(0, self.world_size))
            radius = random.randint(40,80)
            # Add the source to our world
            self.nutrient_field.add_circular_source(pos, radius, 1.0)

    def update(self):
        """
        Update the state of the universe. Each time this function is called, the state of the universe is updated once.
        This is like the heartbeat of the universe, each beat making the universe more complex.
        更新宇宙状态。每次调用这个函数，宇宙的状态都会更新一次。
        这就像是宇宙的心跳，每次心跳都会让宇宙变得更复杂。
        """
        # Currently empty, will add logic later 现在还是空的，以后添加逻辑
        # In the future, we will update the fields here too.
        pass

    def draw(self, screen):
        """
        Draw the state of the universe on the screen.
        This function is called to display the current state of the universe.
        在屏幕上绘制宇宙的状态。
        这个函数会被调用来显示宇宙的当前状态。
        """
        # Now,we tell the nutrient_field to draw itself.
        self.nutrient_field.draw(screen)



# --- 第二部分 主程序入口 ---
def main():
    pygame.init()
    pygame.display.set_caption("涌现认知生态系统 GPU 加速版")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Create a dedicated surface for the simulation
    # Let's make the simulation area a square that fits the screes's height
    SIM_AREA_SIZE = SCREEN_HEIGHT
    sim_surface = pygame.Surface((SIM_AREA_SIZE, SIM_AREA_SIZE))

    universe = Universe()  # 创建宇宙实例

    running = True
    while running:
        #1.事件处理
        #检查所有用户操作
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        #2.更新状态
        universe.update()
        #3.渲染屏幕
        screen.fill((10, 10, 20))
        sim_surface.fill((20,40,80))  # 深蓝色背景
        #让universe把自己画在屏幕上
        universe.draw(sim_surface) # Pass sim_surface, not screen!
        # Finally, blit the finished sim_surface onto the main screen at position (0,0).
        screen.blit(sim_surface, (0,0))
        #更新到屏幕
        pygame.display.flip()

        #控制循环速度，最高60fps
        clock.tick(60)

    pygame.quit()
    print("模拟结束")

if __name__ == "__main__":
    main()