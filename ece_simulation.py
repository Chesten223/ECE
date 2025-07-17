# -*- coding: utf-8 -*-
# =============================================================================
# 涌现认知生态系统 (ECE) - v5.0
#
# 作者: 一个中国的高中复读生 & Claude
# 日期: 2025年7月16日
#
# v5.0 核心功能更新:
# 1. [调整] 实现了"计算的有限深度"法则: 信号传递需要多帧完成，每帧只执行有限计算步骤
# 2. [涌现] 记忆和思想自然涌现自信号传递过程，无需人工设计
# 3. [保留] 封闭能量系统：所有能量均来自模拟开始时的一次性投放
# 4. [保留] 严格碰撞物理：智能体之间不再重叠，实现为硬球模型
# =============================================================================

import pygame
import numpy as np
import random
import math
import os
import datetime
import csv
import json
import time
import multiprocessing
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pygame.math import Vector2
import base64

# --- 第一部分: 宇宙公理 (Axioms of the Universe) ---

# 1. 宇宙设定
INITIAL_SCREEN_WIDTH = 1200 
INITIAL_SCREEN_HEIGHT = 800
WORLD_SIZE = 512
INFO_PANEL_WIDTH = 400

# 2. 演化引擎参数
INITIAL_AGENT_COUNT = 100  # 初始智能体数量设置为100
MAX_AGENTS = 500          # 限制最大智能体数量为500
MIN_AGENTS_TO_SPAWN = 100  # 如果智能体数量低于100，就补充
MUTATION_PROBABILITY = {
    'point': 0.03, 'add_conn': 0.015, 'del_conn': 0.015,
    'add_node': 0.007, 'del_node': 0.007,
}
MUTATION_STRENGTH = 0.2

# 3. 物理与生态参数
FIELD_DIFFUSION_RATE = 0.1
FIELD_DECAY_RATE = 0.001
INTERACTION_RANGE = 120.0 
ENERGY_TRANSFER_EFFICIENCY = 0.9
K_INTERACTION_FACTOR = 0.01
MOVEMENT_SPEED_FACTOR = 50.0  # 增大移动速度因子
MOVEMENT_ENERGY_COST = 0.04   # 略微减少移动能耗
SIGNAL_EMISSION_RADIUS = 20.0 
BIOTIC_FIELD_SPECIAL_DECAY = 2.0
AGENT_RADIUS = 2.0
MILD_REPULSION_RADIUS = 10.0   # 排斥力作用范围
MILD_REPULSION_STRENGTH = 1.2  # 排斥力强度
COLLISION_ITERATIONS = 5       # 碰撞检测迭代次数
HIGH_DENSITY_THRESHOLD = 3     # 高密度区域的邻居数量阈值
OVERLAP_EMERGENCY_DISTANCE = 0.5  # 紧急情况下的额外排斥距离
MIN_MOVEMENT_JITTER = 0.02     # 最小随机移动量，确保所有生物都会动
REPULSION_PRIORITY = 2.0       # 排斥力优先级，确保排斥力优先于神经网络输出
ENERGY_PATCH_RADIUS_MIN = 60.0 # 能量辐射最小范围 (原来是30)
ENERGY_PATCH_RADIUS_MAX = 120.0 # 能量辐射最大范围 (原来是60)
ENERGY_GRADIENT_FACTOR = 0.6   # 能量梯度因子，越小梯度越缓
SPAWN_SAFE_DISTANCE = AGENT_RADIUS * 3.0  # 生成新智能体时的安全距离

# 4. 性能优化参数
MAX_THREADS = max(4, multiprocessing.cpu_count() - 1)  # 使用CPU核心数-1的线程数
BATCH_SIZE = 100  # 每个批次处理的智能体数量
GRID_CELL_SIZE_FACTOR = 1.2  # 网格大小因子，用于空间划分优化
PERFORMANCE_MONITOR = True  # 启用性能监控
UPDATE_INTERVAL = 60  # 性能统计更新间隔（帧数）
RENDER_OPTIMIZATION = True  # 启用渲染优化
FIELD_CACHE_ENABLED = True  # 启用场缓存
COLLISION_OPTIMIZATION = True  # 启用碰撞优化
LOG_BUFFER_SIZE = 1000  # 日志缓冲区大小
LOG_FLUSH_INTERVAL = 5.0  # 日志刷新间隔（秒）
DEFAULT_RENDER_SKIP = 1  # 默认渲染跳过帧数
NEIGHBOR_CACHE_ENABLED = True  # 启用邻居缓存
SPATIAL_GRID_OPTIMIZATION = True  # 启用空间网格优化
CACHE_LIFETIME = 5  # 缓存生命周期（帧数）
AGENT_RENDER_BATCH_SIZE = 50  # 智能体渲染批次大小
USE_SURFACE_CACHING = True  # 使用表面缓存
SIGNAL_RENDER_THRESHOLD = 0.2  # 信号渲染阈值

# --- 数据日志系统 ---
class DataLogger:
    def __init__(self, continue_from=None):
        # 创建新的日志目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("logs", f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        if continue_from:
            # 从指定的日志目录继续模拟，但写入新目录
            self.continue_from_existing = True
            self.source_log_dir = continue_from
            
            # 读取现有日志文件以获取最后一帧和智能体ID计数器
            self.agent_id_counter = self._get_max_agent_id(continue_from)
            self.last_frame = self._get_last_frame(continue_from)
            
            # 复制旧日志文件到新目录
            self._copy_log_files(continue_from)
            
            print(f"继续从日志 {continue_from} 的第 {self.last_frame} 帧开始模拟，新日志保存在 {self.log_dir}")
        else:
            # 全新的模拟
            self.continue_from_existing = False
            self.agent_id_counter = 0
            self.last_frame = 0
        
        # 初始化日志文件路径
        self.state_log_path = os.path.join(self.log_dir, "simulation_log.csv")
        self.event_log_path = os.path.join(self.log_dir, "event_log.csv")
        self.field_log_path = os.path.join(self.log_dir, "field_log.csv")
        self.signal_types_path = os.path.join(self.log_dir, "signal_types.json")
        
        # 如果是全新模拟，创建新文件并写入表头
        if not self.continue_from_existing:
            self.state_header = ["frame", "agent_id", "parent_id", "genotype_id", "is_mutant", "energy", 
                                "pos_x", "pos_y", "n_hidden", "n_connections", "computation_depth", "gene_string"]
            with open(self.state_log_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.state_header)

            self.event_header = ["frame", "event_type", "details"]
            with open(self.event_log_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.event_header)
                
            # 添加场景数据日志文件
            self.field_header = ["frame", "field_type", "data"]
            with open(self.field_log_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.field_header)
        
        # 缓冲区，减少I/O操作
        self.state_buffer = []
        self.event_buffer = []
        self.field_buffer = []  # 新增场数据缓冲区
        self.buffer_size_limit = LOG_BUFFER_SIZE  # 使用全局配置
        self.last_flush_time = time.time()
        self.flush_interval = LOG_FLUSH_INTERVAL  # 使用全局配置
    
    def _copy_log_files(self, source_dir):
        """复制旧日志文件到新目录"""
        try:
            # 复制状态日志
            source_state_log = os.path.join(source_dir, "simulation_log.csv")
            if os.path.exists(source_state_log):
                with open(source_state_log, 'r', newline='', encoding='utf-8') as src, \
                     open(self.state_log_path, 'w', newline='', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            # 复制事件日志
            source_event_log = os.path.join(source_dir, "event_log.csv")
            if os.path.exists(source_event_log):
                with open(source_event_log, 'r', newline='', encoding='utf-8') as src, \
                     open(self.event_log_path, 'w', newline='', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            # 复制场数据日志
            source_field_log = os.path.join(source_dir, "field_log.csv")
            if os.path.exists(source_field_log):
                with open(source_field_log, 'r', newline='', encoding='utf-8') as src, \
                     open(self.field_log_path, 'w', newline='', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            # 复制信号类型日志
            source_signal_types = os.path.join(source_dir, "signal_types.json")
            if os.path.exists(source_signal_types):
                with open(source_signal_types, 'r', encoding='utf-8') as src, \
                     open(self.signal_types_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            print(f"已复制日志文件从 {source_dir} 到 {self.log_dir}")
        except Exception as e:
            print(f"复制日志文件时出错: {str(e)}")
    
    def _get_max_agent_id(self, log_dir=None):
        """从现有日志中获取最大的智能体ID"""
        max_id = 0
        try:
            path = os.path.join(log_dir or self.log_dir, "simulation_log.csv")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) > 1:
                        agent_id = int(row[1])
                        max_id = max(max_id, agent_id)
        except Exception as e:
            print(f"读取智能体ID时出错: {str(e)}")
        return max_id
    
    def _get_last_frame(self, log_dir=None):
        """从现有日志中获取最后一帧的帧号"""
        last_frame = 0
        try:
            path = os.path.join(log_dir or self.log_dir, "simulation_log.csv")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) > 0:
                        frame = int(row[0])
                        last_frame = max(last_frame, frame)
        except Exception as e:
            print(f"读取最后一帧时出错: {str(e)}")
        return last_frame
    
    def load_last_state(self):
        """加载最后一帧的状态，用于恢复模拟"""
        agents_data = []
        try:
            # 如果是继续模拟，从源日志目录读取
            path = os.path.join(self.source_log_dir if hasattr(self, 'source_log_dir') else self.log_dir, "simulation_log.csv")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) > 0 and int(row[0]) == self.last_frame:
                        agents_data.append(row)
        except Exception as e:
            print(f"加载最后状态时出错: {str(e)}")
        return agents_data
    
    def load_signal_types(self):
        """加载信号类型"""
        signal_types = set()
        try:
            # 如果是继续模拟，从源日志目录读取
            path = os.path.join(self.source_log_dir if hasattr(self, 'source_log_dir') else self.log_dir, "signal_types.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    signal_types = set(json.load(f))
        except Exception as e:
            print(f"加载信号类型时出错: {str(e)}")
        return signal_types
    
    def log_signal_types(self, signal_types):
        """记录模拟中出现的信号类型"""
        try:
            with open(self.signal_types_path, 'w', encoding='utf-8') as f:
                json.dump(list(signal_types), f)
        except Exception as e:
            print(f"记录信号类型错误: {str(e)}")
    
    def get_new_agent_id(self):
        self.agent_id_counter += 1
        return self.agent_id_counter

    def log_state(self, frame_number, agents):
        # 将状态信息添加到缓冲区
        for agent in agents:
            gene_str = str(agent.gene)
            row = [frame_number, agent.id, agent.parent_id, agent.genotype_id, agent.is_mutant, 
                  round(agent.energy, 2), round(agent.position.x, 2), round(agent.position.y, 2), 
                  agent.gene['n_hidden'], len(agent.gene['connections']), agent.gene['computation_depth'], gene_str]
            self.state_buffer.append(row)
        
        # 检查是否需要刷新缓冲区
        self._check_flush_buffer()

    def log_event(self, frame, event_type, details):
        # 将事件信息添加到缓冲区
        details_str = json.dumps(details)
        self.event_buffer.append([frame, event_type, details_str])
        
        # 检查是否需要刷新缓冲区
        self._check_flush_buffer()
    
    def log_field(self, frame, fields):
        # 记录场数据
        for idx, field in enumerate(fields):
            # 将numpy数组转换为压缩的base64字符串
            field_data = np.array(field.grid, dtype=np.float32)
            field_bytes = field_data.tobytes()
            encoded_data = base64.b64encode(field_bytes).decode('ascii')
            
            self.field_buffer.append([frame, field.name, encoded_data])
        
        # 检查是否需要刷新缓冲区
        self._check_flush_buffer()
    
    def _check_flush_buffer(self):
        # 如果缓冲区达到大小限制或者距离上次刷新已经过了指定时间，则刷新缓冲区
        current_time = time.time()
        if (len(self.state_buffer) + len(self.event_buffer) + len(self.field_buffer) > self.buffer_size_limit or
            current_time - self.last_flush_time > self.flush_interval):
            self._flush_buffers()
    
    def _flush_buffers(self):
        """刷新缓冲区到文件"""
        try:
            # 刷新状态缓冲区
            if self.state_buffer:
                with open(self.state_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.state_buffer)
                self.state_buffer = []
            
            # 刷新事件缓冲区
            if self.event_buffer:
                with open(self.event_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.event_buffer)
                self.event_buffer = []
                
            # 刷新场数据缓冲区
            if self.field_buffer:
                with open(self.field_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.field_buffer)
                self.field_buffer = []
            
            # 更新最后刷新时间
            self.last_flush_time = time.time()
        except Exception as e:
            print(f"日志刷新错误: {str(e)}")
    
    def __del__(self):
        """确保在对象被销毁时刷新所有缓冲区"""
        try:
            self._flush_buffers()
        except Exception as e:
            print(f"日志销毁时错误: {e}")

# --- 相机系统 ---
class Camera:
    def __init__(self, render_width, render_height):
        self.render_width = render_width
        self.render_height = render_height
        self.zoom = min(render_width, render_height) / WORLD_SIZE
        self.offset = Vector2(WORLD_SIZE / 2, WORLD_SIZE / 2)
        self.panning = False
        self.pan_start_pos = (0, 0)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4 and mouse_pos[0] < self.render_width:
                self.zoom_at(mouse_pos, 1.1)
            elif event.button == 5 and mouse_pos[0] < self.render_width:
                self.zoom_at(mouse_pos, 1/1.1)
            elif event.button == 3:
                self.panning = True
                self.pan_start_pos = mouse_pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3: self.panning = False
        elif event.type == pygame.MOUSEMOTION:
            if self.panning:
                dx, dy = event.pos[0] - self.pan_start_pos[0], event.pos[1] - self.pan_start_pos[1]
                self.offset.x -= dx / self.zoom
                self.offset.y -= dy / self.zoom
                self.pan_start_pos = event.pos

    def zoom_at(self, mouse_pos, scale):
        world_pos_before_zoom = self.screen_to_world(mouse_pos)
        self.zoom *= scale
        world_pos_after_zoom = self.screen_to_world(mouse_pos)
        self.offset += world_pos_before_zoom - world_pos_after_zoom

    def world_to_screen(self, world_pos):
        screen_x = (world_pos.x - self.offset.x) * self.zoom + self.render_width / 2
        screen_y = (world_pos.y - self.offset.y) * self.zoom + self.render_height / 2
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_pos):
        world_x = (screen_pos[0] - self.render_width / 2) / self.zoom + self.offset.x
        world_y = (screen_pos[1] - self.render_height / 2) / self.zoom + self.offset.y
        return Vector2(world_x, world_y)
    
    def update_render_size(self, new_width, new_height):
        self.render_width = new_width
        self.render_height = new_height 

# --- 信息场系统 ---
class Field:
    def __init__(self, size, color, name):
        self.size = size
        self.color = color
        self.name = name
        self.grid = np.zeros((size, size), dtype=np.float32)
        # 缓存上一帧的渲染结果，避免重复计算
        self.last_render_surface = None
        self.last_camera_params = None

    def update(self, dt):
        """更新场 - 使用向量化操作提高性能"""
        # 使用原地操作减少内存分配
        self.grid *= (1.0 - FIELD_DECAY_RATE * dt)
        # 确保场值在0-1范围内
        np.clip(self.grid, 0, 1, out=self.grid)

    def get_value_and_gradient(self, pos):
        """在给定位置获取场值和梯度 - 使用预计算和缓存优化"""
        # 计算整数坐标
        x, y = int(pos.x) % self.size, int(pos.y) % self.size
        
        # 使用预计算的偏移量
        offsets = [
            ((y - 1) % self.size, x),             # 上
            ((y + 1) % self.size, x),             # 下
            (y, (x - 1) % self.size),             # 左
            (y, (x + 1) % self.size)              # 右
        ]
        
        # 批量获取场值
        val_up = self.grid[offsets[0]]
        val_down = self.grid[offsets[1]]
        val_left = self.grid[offsets[2]]
        val_right = self.grid[offsets[3]]
        
        # 计算当前位置的场值
        value = self.grid[y, x]
        
        # 计算梯度
        gradient = Vector2(val_right - val_left, val_down - val_up)
        
        return value, gradient

    def add_circular_source(self, pos, radius, value):
        """在场中添加一个圆形源 - 使用优化的向量化操作"""
        # 如果半径太小，直接返回
        if radius <= 0:
            return
            
        # 转换为整数坐标
        x_center, y_center = int(pos.x), int(pos.y)
        radius = int(radius)
        
        # 计算圆形区域的边界
        x_min = max(0, x_center - radius)
        x_max = min(self.size - 1, x_center + radius)
        y_min = max(0, y_center - radius)
        y_max = min(self.size - 1, y_center + radius)
        
        # 如果边界无效，直接返回
        if x_min >= x_max or y_min >= y_max:
            return
        
        # 创建坐标网格
        y_range = np.arange(y_min, y_max + 1)
        x_range = np.arange(x_min, x_max + 1)
        grid_y, grid_x = np.meshgrid(y_range, x_range, indexing='ij')
        
        # 计算到中心的距离
        dx = grid_x - x_center
        dy = grid_y - y_center
        distances = np.sqrt(dx*dx + dy*dy)
        
        # 创建圆形掩码
        mask = distances <= radius
        
        # 计算梯度值
        gradient_values = np.zeros_like(distances)
        np.multiply(
            value, 
            np.maximum(0, 1 - (distances / radius) ** ENERGY_GRADIENT_FACTOR),
            out=gradient_values, 
            where=mask
        )
        
        # 确保场值不超过1.0
        current_values = self.grid[grid_y, grid_x]
        room_to_add = np.maximum(0, 1.0 - current_values)
        values_to_add = np.minimum(gradient_values, room_to_add)
        
        # 更新场值
        self.grid[grid_y, grid_x] += values_to_add * mask
        
        # 清除渲染缓存
        self.last_render_surface = None

    def draw(self, surface, camera, alpha=128):
        """绘制场 - 使用缓存和视口裁剪优化性能"""
        # 获取当前相机参数
        current_camera_params = (camera.zoom, camera.offset.x, camera.offset.y)
        
        # 检查是否可以重用上一帧的渲染结果
        if (self.last_render_surface is not None and 
            self.last_camera_params == current_camera_params):
            # 直接使用缓存的渲染结果
            surface.blit(self.last_render_surface, (0, 0))
            return
            
        # 保存当前相机参数
        self.last_camera_params = current_camera_params
        
        # 计算视口边界
        render_width, render_height = camera.render_width, camera.render_height
        top_left = camera.screen_to_world((0, 0))
        bottom_right = camera.screen_to_world((render_width, render_height))
        
        # 确保边界在有效范围内
        start_x = max(0, int(top_left.x))
        start_y = max(0, int(top_left.y))
        end_x = min(self.size, int(bottom_right.x) + 2)
        end_y = min(self.size, int(bottom_right.y) + 2)
        
        # 如果视口完全在场外，直接返回
        if start_x >= end_x or start_y >= end_y:
            return
            
        # 提取可见区域的子网格
        sub_grid = self.grid[start_y:end_y, start_x:end_x]
        
        # 如果子网格为空，直接返回
        if sub_grid.size == 0:
            return
            
        # 检查是否有非零值
        if np.max(sub_grid) < 0.01:
            return
        
        # 创建彩色数组
        color_array = np.zeros((sub_grid.shape[1], sub_grid.shape[0], 3), dtype=np.uint8)
        color_array[:, :, self.color] = (sub_grid.T * 255).astype(np.uint8)
        
        # 创建表面并设置透明度
        render_surface = pygame.surfarray.make_surface(color_array)
        render_surface.set_alpha(alpha)
        
        # 计算屏幕位置和缩放尺寸
        screen_pos = camera.world_to_screen(Vector2(start_x, start_y))
        scaled_size = (int(sub_grid.shape[1] * camera.zoom), int(sub_grid.shape[0] * camera.zoom))
        
        # 如果缩放后的尺寸太小，直接返回
        if scaled_size[0] <= 0 or scaled_size[1] <= 0:
            return
            
        # 创建缓存的渲染表面
        self.last_render_surface = pygame.Surface((render_width, render_height), pygame.SRCALPHA)
        
        # 渲染到缓存表面
        scaled_surface = pygame.transform.scale(render_surface, scaled_size)
        self.last_render_surface.blit(scaled_surface, screen_pos)
        
        # 渲染到目标表面
        surface.blit(self.last_render_surface, (0, 0))

# --- 生命单元系统 ---
class Agent:
    def __init__(self, universe, logger, gene=None, position=None, energy=None, parent_id=None, is_mutant=False):
        self.universe = universe
        self.logger = logger
        self.id = self.logger.get_new_agent_id()
        self.parent_id = parent_id
        self.position = position if position else Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE))
        self.energy = energy if energy else 100.0
        self.is_dead = False
        self.is_mutant = is_mutant
        self.genotype_id = None
        
        # 基因初始化
        if gene is None:
            self.gene = self.generate_random_gene()
        else:
            self.gene = gene
            
        # 构建计算核心
        self.build_from_gene()
        
        # 上一次行动向量
        self.last_action_vector = np.zeros(self.gene['n_output'])
        
        # 智能体半径
        self.radius = AGENT_RADIUS

    def generate_random_gene(self):
        # 极简初始化，让系统完全自由演化
        # 简化节点数量
        n_input = random.randint(2, 5)  # 减少初始输入节点数量
        n_output = random.randint(1, 3)  # 减少初始输出节点数量
        n_hidden = random.randint(0, 2)  # 减少初始隐藏节点数量
        
        # 创建极少数的随机连接 - 每个智能体可能只有1-2个连接
        connections = []
        
        # 确保至少有1个连接
        from_node = random.randint(0, n_input - 1)
        to_node = random.randint(n_input, n_input + n_hidden + n_output - 1)
        connections.append([from_node, to_node, random.uniform(-1, 1)])
        
        # 有50%概率添加第二个连接
        if random.random() < 0.5:
            from_node = random.randint(0, n_input + n_hidden - 1)
            to_node = random.randint(n_input, n_input + n_hidden + n_output - 1)
            if to_node > from_node:  # 避免回路
                connections.append([from_node, to_node, random.uniform(-1, 1)])
        
        # 极简的节点类型初始化
        input_types = []
        for _ in range(n_input):
            input_types.append(random.choice(['field_sense', 'signal_sense']))
            
        output_types = []
        for _ in range(n_output):
            output_types.append(random.choice(['movement', 'signal']))
            
        hidden_types = []
        for _ in range(n_hidden):
            hidden_types.append('standard')
        
        # 返回极简的基因结构
        return {
            'n_input': n_input,
            'n_output': n_output, 
            'n_hidden': n_hidden, 
            'computation_depth': random.randint(1, 3),  # 减少初始计算深度
            'connections': connections,
            'env_absorption_coeff': random.uniform(-0.5, 0.5),  # 减少初始吸收系数范围
            'node_types': {
                'input': input_types,
                'output': output_types,
                'hidden': hidden_types
            }
        }

    def build_from_gene(self):
        """从基因构建计算核心 - 使用高效的向量化操作"""
        # 解析基因
        self.n_input = self.gene['n_input']
        self.n_hidden = self.gene['n_hidden']
        self.n_output = self.gene['n_output']
        self.computation_depth = self.gene['computation_depth']
        
        # 计算总节点数
        total_nodes = self.n_input + self.n_hidden + self.n_output
        
        # 初始化节点激活值
        self.node_activations = np.zeros(total_nodes, dtype=np.float32)
        
        # 初始化连接矩阵 - 使用稀疏矩阵表示
        self.connection_matrix = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        
        # 填充连接矩阵
        for from_node, to_node, weight in self.gene['connections']:
            if 0 <= from_node < total_nodes and 0 <= to_node < total_nodes:
                self.connection_matrix[from_node, to_node] = weight
        
        # 预计算常量，减少重复计算
        self.output_start_idx = self.n_input + self.n_hidden
        
        # 初始化上次行动向量
        self.last_action_vector = np.zeros(self.n_output, dtype=np.float32)
        
        # 计算复杂度
        self.complexity = self.n_hidden + len(self.gene['connections']) * 0.5 + self.computation_depth * 0.2
        
        # 设置繁殖参数
        self.e_repro = 20 + self.complexity * 5  # 繁殖阈值随复杂度增加
        self.e_child = 10 + self.complexity * 2  # 子代能量消耗随复杂度增加
        
        # 设置基础代谢成本
        self.metabolism_cost = 0.01 + self.complexity * 0.002  # 代谢成本随复杂度增加
        
        # 环境吸收系数
        self.env_absorption_coeff = self.gene.get('env_absorption_coeff', 0.5)
        
        # 添加身份向量（用于相互作用和生态位定位）
        if self.gene['connections']:
            weights = [c[2] for c in self.gene['connections']]
            self.identity_vector = np.mean(weights)
        else:
            self.identity_vector = 0
        
        # 注册基因型ID
        self.genotype_id = self.universe.get_or_create_genotype_id(self.gene)

    def update(self, dt, neighbors):
        if self.is_dead: 
            return
        
        # 1. 感知与决策 - 实现"计算的有限深度"法则
        # 获取环境感知向量
        perception_vector = self.universe.get_perception_vector(self.position)
        
        # 根据节点类型决定如何处理输入 - 使用向量化操作
        node_types = ['field_sense'] * self.n_input  # 默认类型改为field_sense
        if 'node_types' in self.gene and 'input' in self.gene['node_types']:
            # 获取节点类型，确保长度匹配
            types_list = self.gene['node_types']['input']
            for i in range(min(len(types_list), self.n_input)):
                node_types[i] = types_list[i]
        
        # 批量更新输入节点 - 使用NumPy向量化操作
        # 创建掩码数组以进行批量更新
        field_sense_mask = np.array([t == 'field_sense' for t in node_types[:len(perception_vector)]], dtype=bool)
        signal_sense_mask = np.array([t == 'signal_sense' for t in node_types[:len(perception_vector)]], dtype=bool)
        
        # 批量更新不同类型节点
        if np.any(field_sense_mask) and len(field_sense_mask) == len(perception_vector):
            self.node_activations[:len(perception_vector)][field_sense_mask] = perception_vector[field_sense_mask]
        
        if np.any(signal_sense_mask) and len(signal_sense_mask) == len(perception_vector):
            self.node_activations[:len(perception_vector)][signal_sense_mask] = perception_vector[signal_sense_mask]
        
        # 执行计算步骤（由基因决定的深度）- 使用矩阵运算提高效率
        # 预分配内存以减少重复分配
        new_activations = np.zeros_like(self.node_activations)
        
        # 只计算隐藏层和输出层的激活值，输入层保持不变
        input_activations = self.node_activations[:self.n_input].copy()
        
        for _ in range(self.computation_depth):
            # 计算新的激活值 - 使用矩阵乘法而不是点积，更高效
            inputs = np.matmul(self.node_activations, self.connection_matrix)
            
            # 只更新隐藏层和输出层的激活值
            np.tanh(inputs[self.n_input:], out=new_activations[self.n_input:])
            
            # 更新隐藏层和输出层的激活值，保持输入层不变
            self.node_activations[self.n_input:] = new_activations[self.n_input:]
        
        # 确保输入层不受计算影响
        self.node_activations[:self.n_input] = input_activations
        
        # 读取当前输出层的值作为行动指令
        output_activations = self.node_activations[-self.n_output:]
        self.last_action_vector = output_activations
        
        # 初始化行为向量
        move_vector = Vector2(0, 0)
        
        # 根据输出节点类型决定行为
        for i, activation in enumerate(output_activations):
            # 获取当前输出节点类型
            node_type = 'movement'  # 默认为移动类型
            if 'node_types' in self.gene and 'output' in self.gene['node_types'] and i < len(self.gene['node_types']['output']):
                node_type = self.gene['node_types']['output'][i]
            
            # 根据节点类型执行不同行为
            if node_type == 'movement':
                # 移动节点影响移动向量
                # 每对节点控制一个方向
                if i % 2 == 0 and i+1 < len(output_activations):  # X方向
                    move_vector.x += activation
                elif i % 2 == 1:  # Y方向
                    move_vector.y += activation
            elif node_type == 'signal':
                # 信号节点控制信号释放 - 允许更多信号类型
                # 计算信号场索引 - 允许多达8种不同信号
                signal_count = len(self.universe.fields) - 2  # 减去能量场和危险场
                field_idx = (i % signal_count) + 2  # 从索引2开始(跳过能量场和危险场)
                
                if field_idx < len(self.universe.fields) and abs(activation) > SIGNAL_RENDER_THRESHOLD:
                    # 信号强度与激活值成正比
                    signal_strength = abs(activation) * 0.02
                    
                    # 信号半径与激活值成正比
                    signal_radius = SIGNAL_EMISSION_RADIUS * (0.5 + abs(activation) * 0.5)
                    
                    # 发射信号
                    self.universe.fields[field_idx].add_circular_source(
                        self.position, signal_radius, signal_strength)
                    
                    # 记录信号类型
                    signal_name = f"Signal {field_idx-1}"  # 信号编号从1开始
                    self.universe.signal_types.add(signal_name)
        
        # 确保所有生物都有最小移动量
        if move_vector.length_squared() < MIN_MOVEMENT_JITTER**2:
            move_vector.x += random.uniform(-MIN_MOVEMENT_JITTER, MIN_MOVEMENT_JITTER)
            move_vector.y += random.uniform(-MIN_MOVEMENT_JITTER, MIN_MOVEMENT_JITTER)
        
        # 2. 移动
        self.position += move_vector * dt * MOVEMENT_SPEED_FACTOR

        # 3. 添加温和排斥力 - 使用优化的碰撞检测
        if COLLISION_OPTIMIZATION:
            self._optimized_collision_detection(neighbors, dt)
        else:
            self._standard_collision_detection(neighbors, dt, move_vector)
            
        # 确保在世界边界内
        self.position.x = max(0, min(WORLD_SIZE, self.position.x))
        self.position.y = max(0, min(WORLD_SIZE, self.position.y))

        # 5. 世界边界环绕
        self.position.x %= WORLD_SIZE
        self.position.y %= WORLD_SIZE

        # 6. 与邻近智能体的能量交换（捕食关系）
        for other in neighbors:
            if other is self or other.is_dead: 
                continue
            dist_sq = (self.position - other.position).length_squared()
            if dist_sq < INTERACTION_RANGE**2:
                # 基于身份向量差异的生态位分化捕食关系
                # 计算身份向量差异的绝对值
                identity_diff = abs(self.identity_vector - other.identity_vector)
                
                # 确定最佳捕食差异 - 设为中等差异值时捕食效率最高
                # 过于相似（同类）或过于不同（不兼容的生态位）都降低捕食效率
                OPTIMAL_DIFF = 0.5  # 最佳差异值
                
                # 计算捕食效率 - 使用高斯曲线，在最佳差异处达到峰值
                # 身份差异接近最佳差异时捕食效率最高
                predation_efficiency = math.exp(-10 * (identity_diff - OPTIMAL_DIFF)**2)
                
                # 算法核心：生态位差异适中，并且self身份向量高于other时才能捕食
                # 这确保了捕食是单向的，避免了互相吞噬
                if self.identity_vector > other.identity_vector and predation_efficiency > 0.1:
                    # 距离影响捕食效率
                    dist_factor = 1 - math.sqrt(dist_sq) / INTERACTION_RANGE
                    energy_transfer = predation_efficiency * K_INTERACTION_FACTOR * 30 * dist_factor
                    
                    # 捕食者获得能量，被捕食者失去能量
                    self.energy += energy_transfer * dt
                    other.energy -= energy_transfer * dt
                    
                    # 简化的捕食记录
                    if energy_transfer * dt > 1.0 and random.random() < 0.05:  # 仅记录5%的显著捕食事件，减少日志量
                        self.universe.logger.log_event(
                            self.universe.frame_count, 
                            'PREDATION', 
                            {'pred_id': self.id, 'prey_id': other.id}
                        )

        # 7. 新陈代谢与环境能量吸收
        # 计算行动消耗 - 考虑所有激活节点的成本
        action_cost = move_vector.length_squared() * MOVEMENT_ENERGY_COST
        signal_cost = sum(abs(a) for a in output_activations) * 0.1
        metabolism = self.metabolism_cost + action_cost + signal_cost
        
        # 从环境获取能量
        nutrient_val, _ = self.universe.nutrient_field.get_value_and_gradient(self.position)
        hazard_val, _ = self.universe.hazard_field.get_value_and_gradient(self.position)
        env_gain = self.env_absorption_coeff * nutrient_val * 40
        env_loss = abs(np.tanh(self.identity_vector)) * hazard_val * 30
        
        # 应用能量变化
        self.energy += (env_gain - env_loss - metabolism) * dt

        # 8. 死亡判定
        if self.energy <= 0:
            self.is_dead = True
            self.logger.log_event(self.universe.frame_count, 'AGENT_DEATH', 
                                 {'agent_id': self.id, 'reason': 'energy_depleted'})
            self.universe.on_agent_death(self)
    
    def _optimized_collision_detection(self, neighbors, dt):
        """优化的碰撞检测算法"""
        # 快速预筛选 - 只处理可能发生碰撞的邻居
        potential_colliders = []
        close_neighbors_count = 0
        overlapping_neighbors = 0
        
        # 使用空间哈希快速筛选
        for other in neighbors:
            if other is self: 
                continue
                
            # 快速预检测 - 使用曼哈顿距离作为初步筛选
            dx = min(abs(self.position.x - other.position.x), WORLD_SIZE - abs(self.position.x - other.position.x))
            dy = min(abs(self.position.y - other.position.y), WORLD_SIZE - abs(self.position.y - other.position.y))
            manhattan_dist = dx + dy
            
            # 只处理可能产生碰撞的邻居
            if manhattan_dist < MILD_REPULSION_RADIUS + self.radius + other.radius:
                dist_vec = self.position - other.position
                
                # 处理周期性边界条件
                if dx > WORLD_SIZE / 2:
                    dist_vec.x = -math.copysign(WORLD_SIZE - abs(dist_vec.x), dist_vec.x)
                if dy > WORLD_SIZE / 2:
                    dist_vec.y = -math.copysign(WORLD_SIZE - abs(dist_vec.y), dist_vec.y)
                
                dist_sq = dist_vec.length_squared()
                min_dist = self.radius + other.radius
                
                # 记录邻居数据
                if dist_sq < MILD_REPULSION_RADIUS**2:
                    close_neighbors_count += 1
                    
                    # 只为真正可能重叠的邻居创建详细数据
                    if dist_sq < (min_dist * 1.5)**2:
                        potential_colliders.append({
                            'agent': other,
                            'dist_vec': dist_vec,
                            'dist_sq': dist_sq,
                            'min_dist': min_dist,
                            'is_overlapping': dist_sq < min_dist**2
                        })
                        
                        if dist_sq < min_dist**2:
                            overlapping_neighbors += 1
        
        # 如果没有重叠或接近的邻居，直接返回
        if close_neighbors_count == 0:
            return
            
        # 计算排斥力
        repulsion_vector = Vector2(0, 0)
        for data in potential_colliders:
            dist_sq = data['dist_sq']
            dist_vec = data['dist_vec']
            
            if dist_sq > 1e-6:  # 避免除以零
                # 使用更温和的排斥力计算
                repulsion_strength = 1.0 - (math.sqrt(dist_sq) / MILD_REPULSION_RADIUS)
                # 距离越近，排斥力越强（非线性增强）
                if dist_sq < data['min_dist']**2:
                    repulsion_strength *= 2.0  # 重叠时加倍排斥力
                repulsion_vector += dist_vec.normalize() * repulsion_strength
        
        # 应用排斥力
        if repulsion_vector.length_squared() > 0:
            density_factor = 1.0
            if close_neighbors_count > HIGH_DENSITY_THRESHOLD:
                # 高密度区域增强排斥力 - 使用非线性增强
                density_factor = 1.0 + (close_neighbors_count - HIGH_DENSITY_THRESHOLD) ** 1.5 * 0.1
            
            # 应用排斥力
            self.position += repulsion_vector * MILD_REPULSION_STRENGTH * density_factor * dt * REPULSION_PRIORITY
        
        # 只有在有重叠时才进行碰撞解决
        if overlapping_neighbors > 0:
            # 减少迭代次数以提高性能
            max_iterations = min(COLLISION_ITERATIONS, 1 + overlapping_neighbors)
            
            # 按距离排序，先处理最严重的重叠
            sorted_colliders = sorted(
                [c for c in potential_colliders if c['is_overlapping']], 
                key=lambda x: x['dist_sq']
            )
            
            # 只进行一次迭代，处理最严重的重叠
            for data in sorted_colliders:
                if data['dist_sq'] < data['min_dist']**2 and data['dist_sq'] > 0:
                    overlap = data['min_dist'] - math.sqrt(data['dist_sq'])
                    # 将当前智能体沿碰撞向量推开整个重叠距离
                    push_vector = data['dist_vec'].normalize() * overlap
                    self.position += push_vector

    def _standard_collision_detection(self, neighbors, dt, move_vector):
        """标准的碰撞检测算法（原始版本）"""
        # 3. 添加温和排斥力
        repulsion_vector = Vector2(0, 0)
        close_neighbors_count = 0
        overlapping_neighbors = 0
        
        # 收集所有邻居信息，以便更好地处理重叠 - 使用向量化操作
        neighbor_data = []
        
        # 预先筛选可能产生碰撞的邻居，减少后续计算量
        potential_colliders = []
        for other in neighbors:
            if other is self: 
                continue
                
            # 快速预检测 - 使用曼哈顿距离作为初步筛选
            dx = min(abs(self.position.x - other.position.x), WORLD_SIZE - abs(self.position.x - other.position.x))
            dy = min(abs(self.position.y - other.position.y), WORLD_SIZE - abs(self.position.y - other.position.y))
            manhattan_dist = dx + dy
            
            # 只有可能产生碰撞的邻居才进行详细计算
            if manhattan_dist < MILD_REPULSION_RADIUS + self.radius + other.radius:
                dist_vec = self.position - other.position
                
                # 处理周期性边界条件
                if dx > WORLD_SIZE / 2:
                    dist_vec.x = -math.copysign(WORLD_SIZE - abs(dist_vec.x), dist_vec.x)
                if dy > WORLD_SIZE / 2:
                    dist_vec.y = -math.copysign(WORLD_SIZE - abs(dist_vec.y), dist_vec.y)
                
                dist_sq = dist_vec.length_squared()
                min_dist = self.radius + other.radius
                
                # 收集邻居数据
                potential_colliders.append({
                    'agent': other,
                    'dist_vec': dist_vec,
                    'dist_sq': dist_sq,
                    'min_dist': min_dist,
                    'is_overlapping': dist_sq < min_dist**2
                })
                
                # 检测是否有重叠
                if dist_sq < min_dist**2:
                    overlapping_neighbors += 1
                
                if dist_sq < MILD_REPULSION_RADIUS**2:
                    close_neighbors_count += 1
                    if dist_sq > 1e-6:  # 避免除以零
                        # 使用更温和的排斥力计算
                        repulsion_strength = 1.0 - (math.sqrt(dist_sq) / MILD_REPULSION_RADIUS)
                        # 距离越近，排斥力越强（非线性增强）
                        if dist_sq < min_dist**2:
                            repulsion_strength *= 2.0  # 重叠时加倍排斥力
                        repulsion_vector += dist_vec.normalize() * repulsion_strength
        
        # 应用温和排斥力，高密度区域增强排斥
        if close_neighbors_count > 0:
            density_factor = 1.0
            if close_neighbors_count > HIGH_DENSITY_THRESHOLD:
                # 高密度区域增强排斥力 - 使用非线性增强
                density_factor = 1.0 + (close_neighbors_count - HIGH_DENSITY_THRESHOLD) ** 1.5 * 0.1
            
            # 确保排斥力优先于神经网络的移动决策
            repulsion_move = repulsion_vector * MILD_REPULSION_STRENGTH * density_factor * dt
            
            # 如果排斥力和移动向量方向相反，优先考虑排斥力
            if move_vector.dot(repulsion_vector) < 0:
                # 当排斥力和移动向量冲突时，增强排斥力的影响
                self.position += repulsion_move * REPULSION_PRIORITY
            else:
                self.position += repulsion_move
        
        # 紧急处理严重重叠情况 - 对于高密度区域增强处理
        if overlapping_neighbors > 0:  # 只要有重叠就处理
            # 计算远离所有重叠邻居的方向
            escape_vector = Vector2(0, 0)
            for data in potential_colliders:
                if data['is_overlapping']:
                    # 距离越近，逃离力越强
                    escape_strength = 1.0
                    if data['dist_sq'] > 0:
                        escape_strength = min(3.0, (data['min_dist']**2) / data['dist_sq'])
                    escape_vector += data['dist_vec'].normalize() * escape_strength
            
            if escape_vector.length_squared() > 0:
                escape_factor = min(1.0, overlapping_neighbors * 0.3)  # 重叠越多，逃离越强
                escape_vector = escape_vector.normalize() * OVERLAP_EMERGENCY_DISTANCE * escape_factor
                self.position += escape_vector
        
        # 4. 严格碰撞解决 - 减少迭代次数，优化计算
        # 只有当存在重叠时才执行碰撞解决
        if overlapping_neighbors > 0:
            # 减少迭代次数以提高性能
            max_iterations = min(COLLISION_ITERATIONS, 1 + overlapping_neighbors)
            
            for iteration in range(max_iterations):
                collision_occurred = False
                # 按距离排序，先处理最严重的重叠
                sorted_colliders = sorted(
                    [c for c in potential_colliders if c['is_overlapping']], 
                    key=lambda x: x['dist_sq']
                )
                
                for data in sorted_colliders:
                    if data['dist_sq'] < data['min_dist']**2 and data['dist_sq'] > 0:
                        collision_occurred = True
                        overlap = data['min_dist'] - math.sqrt(data['dist_sq'])
                        # 将当前智能体沿碰撞向量推开整个重叠距离
                        push_factor = 1.0 + iteration * 0.2  # 每次迭代增加20%的推力
                        
                        # 对于严重重叠，增加额外推力
                        if overlap > data['min_dist'] * 0.5:  # 如果重叠超过半径
                            push_factor *= 1.5
                        
                        # 在高密度区域增加额外推力
                        if close_neighbors_count > HIGH_DENSITY_THRESHOLD:
                            push_factor *= (1.0 + (close_neighbors_count - HIGH_DENSITY_THRESHOLD) * 0.1)
                            
                        self.position += data['dist_vec'].normalize() * overlap * push_factor * (1.0 / max_iterations)
                
                # 如果没有碰撞发生，提前退出循环
                if not collision_occurred:
                    break

    def reproduce(self):
        # 繁殖检查：能量必须达到繁殖阈值
        if self.energy < self.e_repro:
            return None

        # 使用空间网格获取周围智能体，而不是检查所有智能体
        neighbors = self.universe.get_neighbors(self)
        
        # 尝试找到一个没有重叠的位置
        max_attempts = 30  # 增加尝试次数，从20增加到30
        child_pos = None
        min_safe_distance = self.radius * 2.5  # 降低安全距离要求，从3.0降低到2.5
        
        # 缓存邻居位置 - 只检查邻近区域而不是全部智能体
        neighbor_positions = []
        for agent in neighbors:
            if agent is not self and not agent.is_dead:
                neighbor_positions.append(agent.position)
        
        for attempt in range(max_attempts):
            # 生成一个候选位置
            angle = random.uniform(0, 2 * math.pi)
            # 随着尝试次数增加，逐渐扩大搜索范围
            distance_factor = 1.0 + attempt * 0.1
            distance = random.uniform(self.radius * 2.0, self.radius * 10.0 * distance_factor)
            candidate_pos = Vector2(
                self.position.x + math.cos(angle) * distance,
                self.position.y + math.sin(angle) * distance
            )
            
            # 对周期性边界条件进行修正
            candidate_pos.x %= WORLD_SIZE
            candidate_pos.y %= WORLD_SIZE
            
            # 检查这个位置是否会与邻近智能体重叠
            is_valid = True
            for pos in neighbor_positions:
                # 考虑周期性边界条件计算距离
                dx = min(abs(candidate_pos.x - pos.x), WORLD_SIZE - abs(candidate_pos.x - pos.x))
                dy = min(abs(candidate_pos.y - pos.y), WORLD_SIZE - abs(candidate_pos.y - pos.y))
                dist_sq = dx * dx + dy * dy
                
                if dist_sq < min_safe_distance * min_safe_distance:
                    is_valid = False
                    break
            
            if is_valid:
                child_pos = candidate_pos
                break
        
        # 如果找不到合适的位置，则不繁殖，但记录这个事件
        if child_pos is None:
            # 记录繁殖失败事件
            self.logger.log_event(self.universe.frame_count, 'REPRODUCTION_FAILED', 
                                 {'agent_id': self.id, 'reason': 'no_valid_position', 
                                  'neighbors': len(neighbor_positions)})
            return None

        # 增加繁殖的额外能量消耗
        # 基础繁殖成本
        reproduction_cost = self.e_child
        # 额外繁殖开销 - 比例为总繁殖能量的20%
        extra_cost = self.e_child * 0.2
        total_cost = reproduction_cost + extra_cost
        
        # 消耗能量创建后代
        self.energy -= total_cost
        
        # 将能量分配给子代 - 只分配基础繁殖成本
        child_energy = reproduction_cost
        
        # 复制基因并可能发生突变
        new_gene = json.loads(json.dumps(self.gene))
        mutations_occurred = []

        # ===== 基因连接突变 =====
        # 点突变：调整连接权重
        for conn in new_gene['connections']:
            if random.random() < MUTATION_PROBABILITY['point']:
                conn[2] += random.uniform(-1, 1) * MUTATION_STRENGTH
                mutations_occurred.append('point_mutation')
                
        # 添加连接
        if random.random() < MUTATION_PROBABILITY['add_conn']:
            # 确保有足够的节点可以添加连接
            total_nodes = new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']
            if new_gene['n_input'] > 0 and total_nodes > new_gene['n_input']:  # 确保有源节点和目标节点
                from_n = random.randint(0, new_gene['n_input'] + new_gene['n_hidden'] - 1)
                to_n = random.randint(new_gene['n_input'], total_nodes - 1)
                if to_n > from_n:  # 避免回路
                    new_gene['connections'].append([from_n, to_n, random.uniform(-1, 1)])
                    mutations_occurred.append('add_connection')
            
        # 删除连接
        if random.random() < MUTATION_PROBABILITY['del_conn'] and len(new_gene['connections']) > 0:
            # 只有在有连接可删除时才删除
            new_gene['connections'].pop(random.randrange(len(new_gene['connections'])))
            mutations_occurred.append('delete_connection')
            
        # ===== 神经网络参数突变 =====
        # 环境吸收系数突变
        if 'env_absorption_coeff' in new_gene and random.random() < MUTATION_PROBABILITY['point']:
            new_gene['env_absorption_coeff'] += random.uniform(-1, 1) * MUTATION_STRENGTH
            mutations_occurred.append('absorption_coeff_mutation')
            
        # 计算深度突变
        if random.random() < MUTATION_PROBABILITY['point']:
            depth_change = random.choice([-1, 1])
            new_depth = max(1, min(10, new_gene['computation_depth'] + depth_change))
            new_gene['computation_depth'] = new_depth
            mutations_occurred.append('computation_depth_mutation')
        
        # ===== 节点突变 =====
        
        # 1. 添加输入节点突变
        if random.random() < MUTATION_PROBABILITY['add_node'] * 0.5:
            # 完全随机添加输入节点，不限制最大数量
            new_gene['n_input'] += 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                # 随机选择新节点类型
                new_type = random.choice(['field_sense', 'signal_sense'])
                new_gene['node_types']['input'].append(new_type)
            
            # 为新节点创建随机连接
            total_nodes = new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']
            if total_nodes > new_gene['n_input']:  # 确保有目标节点可连接
                for _ in range(random.randint(1, 3)):
                    to_node = random.randint(new_gene['n_input'], 
                                            total_nodes - 1)
                    new_gene['connections'].append([new_gene['n_input'] - 1, to_node, random.uniform(-2, 2)])
            
            mutations_occurred.append('add_input_node')
        
        # 2. 删除输入节点突变
        if random.random() < MUTATION_PROBABILITY['del_node'] * 0.5 and new_gene['n_input'] > 0:
            # 允许删除所有输入节点，不再保留最小数量
            del_node_idx = random.randint(0, new_gene['n_input'] - 1)
            
            # 删除与此节点相关的所有连接
            new_gene['connections'] = [c for c in new_gene['connections'] if c[0] != del_node_idx]
            
            # 更新所有大于删除节点索引的连接
            for conn in new_gene['connections']:
                if conn[0] > del_node_idx:
                    conn[0] -= 1
                if conn[1] > del_node_idx:
                    conn[1] -= 1
            
            # 更新节点数量
            new_gene['n_input'] -= 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                new_gene['node_types']['input'].pop(del_node_idx)
            
            mutations_occurred.append('delete_input_node')
        
        # 3. 添加输出节点突变
        if random.random() < MUTATION_PROBABILITY['add_node'] * 0.5:
            # 完全随机添加输出节点，不限制最大数量
            # 更新节点索引计算
            output_start = new_gene['n_input'] + new_gene['n_hidden']
            new_output_idx = output_start + new_gene['n_output']
            
            # 为新输出节点创建随机连接
            if output_start > 0:  # 确保有源节点可连接
                for _ in range(random.randint(1, 3)):
                    from_node = random.randint(0, output_start - 1)
                    new_gene['connections'].append([from_node, new_output_idx, random.uniform(-2, 2)])
            
            # 更新节点数量
            new_gene['n_output'] += 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                # 随机选择新节点类型
                new_type = random.choice(['movement', 'signal'])
                new_gene['node_types']['output'].append(new_type)
            
            mutations_occurred.append('add_output_node')
        
        # 4. 删除输出节点突变
        if random.random() < MUTATION_PROBABILITY['del_node'] * 0.5 and new_gene['n_output'] > 0:
            # 允许删除所有输出节点，不再保留最小数量
            # 计算要删除的节点索引
            output_start = new_gene['n_input'] + new_gene['n_hidden']
            del_node_idx = output_start + random.randint(0, new_gene['n_output'] - 1)
            
            # 删除与此节点相关的所有连接
            new_gene['connections'] = [c for c in new_gene['connections'] if c[1] != del_node_idx]
            
            # 更新所有大于删除节点索引的连接
            for conn in new_gene['connections']:
                if conn[1] > del_node_idx:
                    conn[1] -= 1
            
            # 更新节点数量
            new_gene['n_output'] -= 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                del_idx = del_node_idx - output_start
                if 0 <= del_idx < len(new_gene['node_types']['output']):
                    new_gene['node_types']['output'].pop(del_idx)
            
            mutations_occurred.append('delete_output_node')
        
        # 5. 添加隐藏节点
        if random.random() < MUTATION_PROBABILITY['add_node']:
            # 添加新的隐藏节点
            hidden_start = new_gene['n_input']
            new_hidden_idx = hidden_start + new_gene['n_hidden']
            
            # 为新隐藏节点创建输入和输出连接
            # 输入连接
            if hidden_start > 0:  # 确保有源节点可连接
                from_node = random.randint(0, hidden_start - 1)
                new_gene['connections'].append([from_node, new_hidden_idx, random.uniform(-2, 2)])
            
            # 输出连接
            total_nodes = new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']
            if new_hidden_idx + 1 < total_nodes:  # 确保有目标节点可连接
                to_node = random.randint(new_hidden_idx + 1, total_nodes - 1)
                new_gene['connections'].append([new_hidden_idx, to_node, random.uniform(-2, 2)])
            
            # 更新节点数量
            new_gene['n_hidden'] += 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                # 所有隐藏节点都是标准类型
                new_gene['node_types']['hidden'].append('standard')
            
            mutations_occurred.append('add_hidden_node')
            
        # 6. 删除隐藏节点
        if new_gene['n_hidden'] > 0 and random.random() < MUTATION_PROBABILITY['del_node']:
            # 随机选择要删除的隐藏节点
            hidden_start = new_gene['n_input']
            del_node_idx = hidden_start + random.randint(0, new_gene['n_hidden'] - 1)
            
            # 删除与此节点相关的所有连接
            new_gene['connections'] = [c for c in new_gene['connections'] 
                                      if c[0] != del_node_idx and c[1] != del_node_idx]
            
            # 更新所有大于删除节点索引的连接
            for conn in new_gene['connections']:
                if conn[0] > del_node_idx:
                    conn[0] -= 1
                if conn[1] > del_node_idx:
                    conn[1] -= 1
            
            # 更新节点数量
            new_gene['n_hidden'] -= 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                del_idx = del_node_idx - hidden_start
                if 0 <= del_idx < len(new_gene['node_types']['hidden']):
                    new_gene['node_types']['hidden'].pop(del_idx)
            
            mutations_occurred.append('delete_hidden_node')

        # 7. 节点类型突变
        if 'node_types' in new_gene and random.random() < MUTATION_PROBABILITY['point'] * 0.5:
            # 随机选择一个节点类别
            valid_categories = []
            for category in ['input', 'output', 'hidden']:
                if category in new_gene['node_types'] and len(new_gene['node_types'][category]) > 0:
                    valid_categories.append(category)
            
            if valid_categories:  # 只有在有有效类别时才进行突变
                node_category = random.choice(valid_categories)
                # 随机选择该类别中的一个节点
                node_idx = random.randint(0, len(new_gene['node_types'][node_category]) - 1)
                
                # 根据类别提供不同的可能类型
                if node_category == 'input':
                    new_type = random.choice(['field_sense', 'signal_sense'])
                elif node_category == 'output':
                    new_type = random.choice(['movement', 'signal'])
                else:  # hidden
                    new_type = 'standard'  # 隐藏节点只有标准类型
                
                # 应用新类型
                new_gene['node_types'][node_category][node_idx] = new_type
                mutations_occurred.append('node_type_mutation')

        # 检查是否发生了突变
        is_mutant = len(mutations_occurred) > 0
        
        # 创建子代（使用找到的无重叠位置）
        child = Agent(self.universe, self.logger, gene=new_gene, position=child_pos, 
                     energy=child_energy, parent_id=self.id, is_mutant=is_mutant)
        
        # 将繁殖和突变事件记录到日志
        if is_mutant:
            self.logger.log_event(
                self.universe.frame_count, 'MUTATION', 
                {'parent_id': self.id, 'child_id': child.id, 'mutations': mutations_occurred}
            )
        
        return child

    def draw(self, surface, camera):
        """绘制智能体 - 使用批量渲染和视口裁剪优化"""
        # 在无GUI模式或智能体已死亡时跳过渲染
        if not self.universe.use_gui or self.is_dead:
            return
            
        # 检查智能体是否在视口内
        screen_pos = camera.world_to_screen(self.position)
        if (screen_pos[0] < -50 or screen_pos[0] > camera.render_width + 50 or
            screen_pos[1] < -50 or screen_pos[1] > camera.render_height + 50):
            return  # 如果不在视口内，不绘制
        
        # 计算屏幕上的半径
        radius = max(1, int(self.radius * camera.zoom))
        
        # 基于基因型ID设置颜色 - 恢复原有的颜色区分
        hue = (self.genotype_id * 20) % 360
        color = pygame.Color(0)
        color.hsva = (hue, 85, 90, 100)
        
        # 根据能量水平调整亮度
        energy_ratio = min(1.0, self.energy / max(0.1, self.e_repro))
        if energy_ratio < 0.3:
            # 能量不足时颜色变暗
            _, s, v, _ = color.hsva
            color.hsva = (hue, s, max(30, int(v * energy_ratio / 0.3)), 100)
            
        # 绘制智能体主体
        pygame.draw.circle(surface, color, screen_pos, radius)
        
        # 如果是被选中的智能体，绘制选中标记
        if self.universe.selected_agent is self:
            # 绘制选中标记 - 使用更高效的方式
            highlight_radius = radius + 3
            pygame.draw.circle(surface, (255, 255, 255), screen_pos, highlight_radius, 1)
            
            # 只为选中的智能体绘制详细信息
            # 绘制身份标记
            id_color = (255, 200, 100) if self.is_mutant else (100, 200, 255)
            id_text = str(self.id)
            
            # 使用预渲染文本而不是每帧重新渲染
            font = pygame.font.SysFont(None, 14)
            text_surface = font.render(id_text, True, id_color)
            text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - radius - 10))
            surface.blit(text_surface, text_rect)
            
            # 绘制信号发射
            base_output = self.gene.get('base_output', 2)
            for i, signal in enumerate(self.last_action_vector):
                if abs(signal) > 0.2:  # 只显示强度足够的信号
                    signal_radius = int(SIGNAL_EMISSION_RADIUS * abs(signal) * camera.zoom)
                    if signal_radius > 0:
                        # 为不同信号使用不同颜色
                        signal_color = (200, 0, 0, 50) if i == 0 else (0, 0, 200, 50) if i == 1 else (0, 200, 0, 50)
                        # 创建透明表面来绘制信号
                        signal_surface = pygame.Surface((signal_radius*2, signal_radius*2), pygame.SRCALPHA)
                        pygame.draw.circle(signal_surface, signal_color, (signal_radius, signal_radius), signal_radius)
                        # 绘制到主表面
                        surface.blit(signal_surface, (screen_pos[0] - signal_radius, screen_pos[1] - signal_radius))
        else:
            # 非选中智能体绘制简单轮廓
            if radius <= 2:
                pygame.draw.circle(surface, color, screen_pos, 1)
            else:
                pygame.draw.circle(surface, color, screen_pos, radius, 1)

# --- 宇宙系统 ---
class Universe:
    def __init__(self, logger, render_width, render_height, use_gui=True, continue_simulation=False):
        self.logger = logger
        self.use_gui = use_gui
        self.continue_simulation = continue_simulation
        
        # 初始化信息场
        self.fields = [
            Field(WORLD_SIZE, 1, "Nutrient/Energy"),  # 营养/能量场（绿色）
            Field(WORLD_SIZE, 0, "Hazard"),          # 危险/障碍场（红色）
            Field(WORLD_SIZE, 2, "Biotic 1"),        # 生物信号场1（蓝色）
            Field(WORLD_SIZE, 0, "Biotic 2"),        # 生物信号场2（红色）
        ]
        self.nutrient_field, self.hazard_field, self.biotic_field_1, self.biotic_field_2 = self.fields
        
        # 跟踪出现的信号类型
        self.signal_types = set()
        
        # 初始化宇宙状态
        self.frame_count = 0 if not continue_simulation else logger.last_frame
        self.selected_agent = None
        self.view_mode = 1  # 默认显示营养场
        
        # 初始化相机（仅在GUI模式下使用）
        if self.use_gui:
            self.camera = Camera(render_width, render_height)
        
        # 初始化空间网格（用于邻居查找优化）
        self.grid_cell_size = INTERACTION_RANGE * GRID_CELL_SIZE_FACTOR
        self.spatial_grid = defaultdict(list)
        
        # 添加邻居缓存系统
        self.neighbor_cache = {}  # 智能体ID到邻居列表的映射
        self.neighbor_cache_frame = {}  # 智能体ID到缓存创建帧的映射
        self.grid_coords_cache = {}  # 预计算的网格坐标缓存
        
        # 基因型注册表
        self.genotype_registry = {}
        self.next_genotype_id = 0
        
        # 封闭能量系统：在模拟开始时一次性投放能量
        self._initial_energy_seeding()
        
        # 创建初始智能体
        self.agents = []
        
        if continue_simulation:
            # 从日志加载智能体
            self._load_agents_from_log()
        else:
            # 创建新的智能体
            self._create_initial_agents()
        
        # 记录实际创建的智能体数量
        actual_count = len(self.agents)
        if actual_count < INITIAL_AGENT_COUNT:
            self.logger.log_event(0, 'SPAWN_WARNING', 
                                {'message': f'Only created {actual_count}/{INITIAL_AGENT_COUNT} agents due to space constraints'})
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        
        # 性能监控
        self.perf_monitor = PerformanceMonitor() if PERFORMANCE_MONITOR else None
        
        # 预计算网格坐标偏移量
        self._precompute_grid_offsets()
    
    def _precompute_grid_offsets(self):
        """预计算网格坐标偏移量"""
        self.grid_offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),  (0, 0),  (1, 0),
            (-1, 1),  (0, 1),  (1, 1)
        ]

    def _initial_energy_seeding(self):
        """在世界中一次性播种初始能量。"""
        num_patches = 5  # 减少为5个能量原点
        for _ in range(num_patches):
            pos = Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE))
            radius = random.uniform(ENERGY_PATCH_RADIUS_MIN, ENERGY_PATCH_RADIUS_MAX)
            self.nutrient_field.add_circular_source(pos, radius, 1.0)
        self.logger.log_event(0, 'INITIAL_ENERGY_SEED', {'patches': num_patches})

    def _get_canonical_gene(self, gene):
        """将基因字典转换为可哈希的、唯一的表示形式"""
        # 处理连接，转换为可哈希的元组
        sorted_connections = tuple(sorted(tuple(c) for c in gene['connections']))
        
        # 创建基因项目的列表
        canonical_items = []
        
        # 处理普通值（非字典）
        for k, v in gene.items():
            if k == 'connections':
                canonical_items.append((k, sorted_connections))
            elif k == 'node_types':
                # 处理node_types字典，将其转换为可哈希的形式
                node_types_tuple = ()
                for type_key, type_values in v.items():
                    # 确保类型值是可哈希的（转换为元组）
                    node_types_tuple += ((type_key, tuple(type_values)),)
                canonical_items.append((k, node_types_tuple))
            else:
                canonical_items.append((k, v))
        
        # 排序并转换为元组，确保可哈希
        return tuple(sorted(canonical_items))

    def get_or_create_genotype_id(self, gene):
        """获取或创建一个新的基因型ID"""
        canonical_gene = self._get_canonical_gene(gene)
        if canonical_gene not in self.genotype_registry:
            self.genotype_registry[canonical_gene] = self.next_genotype_id
            self.next_genotype_id += 1
        return self.genotype_registry[canonical_gene]

    def get_perception_vector(self, pos):
        """获取给定位置的感知向量（所有场的值和梯度）"""
        perception = []
        for field in self.fields:
            val, grad = field.get_value_and_gradient(pos)
            perception.extend([val, grad.x, grad.y])
        return np.array(perception, dtype=np.float32)
    
    def get_perception_vector_template(self):
        """获取感知向量模板（用于确定输入维度）"""
        return np.zeros(len(self.fields) * 3)
    
    def on_agent_death(self, agent):
        """处理智能体死亡事件"""
        # 移除死亡时释放残余能量回馈到环境中的功能
        pass

    def update_spatial_grid(self):
        """更新空间网格（用于邻居查找）- 使用更高效的实现"""
        # 清空网格但保留字典结构以减少内存分配
        if SPATIAL_GRID_OPTIMIZATION:
            # 只清空有智能体的网格单元
            for key in list(self.spatial_grid.keys()):
                self.spatial_grid[key] = []
        else:
            # 传统方式：清空整个网格
            self.spatial_grid.clear()
        
        # 使用批量处理而不是逐个添加
        grid_assignments = {}  # 临时存储网格分配
        
        # 第一步：计算每个智能体所在的网格单元
        for agent in self.agents:
            if not agent.is_dead:
                grid_x = int(agent.position.x / self.grid_cell_size)
                grid_y = int(agent.position.y / self.grid_cell_size)
                grid_key = (grid_x, grid_y)
                
                # 将智能体添加到对应的网格单元列表
                if grid_key not in grid_assignments:
                    grid_assignments[grid_key] = []
                grid_assignments[grid_key].append(agent)
                
                # 更新智能体的网格坐标缓存
                if NEIGHBOR_CACHE_ENABLED:
                    self.grid_coords_cache[agent.id] = grid_key
        
        # 第二步：批量更新空间网格
        for grid_key, agents in grid_assignments.items():
            self.spatial_grid[grid_key] = agents
        
        # 清除邻居缓存
        if NEIGHBOR_CACHE_ENABLED:
            self.neighbor_cache.clear()
            self.neighbor_cache_frame.clear()

    def get_neighbors(self, agent):
        """获取智能体的邻居（包括自身）- 使用缓存和预计算优化"""
        # 检查缓存
        if NEIGHBOR_CACHE_ENABLED:
            # 如果有有效缓存，直接返回
            if agent.id in self.neighbor_cache:
                cache_frame = self.neighbor_cache_frame.get(agent.id, 0)
                if self.frame_count - cache_frame <= CACHE_LIFETIME:
                    return self.neighbor_cache[agent.id]
            
            # 如果智能体位置已经在网格坐标缓存中
            if agent.id in self.grid_coords_cache:
                grid_x, grid_y = self.grid_coords_cache[agent.id]
            else:
                grid_x = int(agent.position.x / self.grid_cell_size)
                grid_y = int(agent.position.y / self.grid_cell_size)
                self.grid_coords_cache[agent.id] = (grid_x, grid_y)
        else:
            # 不使用缓存，每次计算网格坐标
            grid_x = int(agent.position.x / self.grid_cell_size)
            grid_y = int(agent.position.y / self.grid_cell_size)
        
        grid_w = int(WORLD_SIZE / self.grid_cell_size)
        grid_h = int(WORLD_SIZE / self.grid_cell_size)
        
        # 使用列表推导式一次性获取所有邻居，减少循环开销
        neighbors = []
        
        # 使用预计算的网格偏移量
        for dx, dy in self.grid_offsets:
            wrapped_x = (grid_x + dx) % grid_w
            wrapped_y = (grid_y + dy) % grid_h
            grid_key = (wrapped_x, wrapped_y)
            if grid_key in self.spatial_grid:
                neighbors.extend(self.spatial_grid[grid_key])
        
        # 更新缓存
        if NEIGHBOR_CACHE_ENABLED:
            self.neighbor_cache[agent.id] = neighbors
            self.neighbor_cache_frame[agent.id] = self.frame_count
                
        return neighbors

    def _update_agent_batch(self, agent_batch, dt):
        """更新一批智能体（用于并行处理）"""
        for agent in agent_batch:
            if not agent.is_dead:
                neighbors = self.get_neighbors(agent)
                agent.update(dt, neighbors)
        return [agent for agent in agent_batch if not agent.is_dead]

    def _process_reproduction(self, agents):
        """处理一批智能体的繁殖（用于并行处理）"""
        new_children = []
        for agent in agents:
            if not agent.is_dead:
                child = agent.reproduce()
                if child:
                    new_children.append(child)
        return new_children

    def _update_fields_parallel(self, dt):
        """并行更新所有场"""
        futures = []
        for field in self.fields:
            futures.append(self.thread_pool.submit(field.update, dt))
        
        # 等待所有场更新完成
        for future in futures:
            future.result()
        
        # 生物场的特殊衰减（信号更快消失）- 使用向量化操作
        self.biotic_field_1.grid *= (1 - BIOTIC_FIELD_SPECIAL_DECAY * dt)
        self.biotic_field_2.grid *= (1 - BIOTIC_FIELD_SPECIAL_DECAY * dt)

    def _spawn_new_agents(self):
        """生成新的随机智能体"""
        # 计算需要添加的智能体数量，确保达到最小数量
        agents_to_add = MIN_AGENTS_TO_SPAWN - len(self.agents)
        if agents_to_add <= 0:
            return
            
        self.logger.log_event(self.frame_count, 'SPAWN_NEW', 
                             {'count': agents_to_add, 'reason': 'below_minimum'})
        
        new_agents = []
        for _ in range(agents_to_add):
            # 尝试找到一个不重叠的位置
            max_attempts = 30  # 每个智能体尝试位置的最大次数
            new_pos = None
            min_safe_distance = AGENT_RADIUS * 3.0  # 安全距离
            
            # 缓存所有现有智能体位置
            existing_positions = [agent.position for agent in self.agents if not agent.is_dead]
            
            for _ in range(max_attempts):
                candidate_pos = Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE))
                
                # 检查是否与现有智能体重叠
                is_valid = True
                for pos in existing_positions:
                    # 考虑周期性边界条件计算距离
                    dx = min(abs(candidate_pos.x - pos.x), WORLD_SIZE - abs(candidate_pos.x - pos.x))
                    dy = min(abs(candidate_pos.y - pos.y), WORLD_SIZE - abs(candidate_pos.y - pos.y))
                    dist_sq = dx * dx + dy * dy
                    
                    if dist_sq < min_safe_distance * min_safe_distance:
                        is_valid = False
                        break
                
                if is_valid:
                    new_pos = candidate_pos
                    break
            
            # 只有找到合适位置才创建新智能体
            if new_pos:
                new_agents.append(Agent(self, self.logger, position=new_pos))
        
        self.agents.extend(new_agents)
        
        # 记录实际添加的智能体数量
        if len(new_agents) < agents_to_add:
            self.logger.log_event(self.frame_count, 'SPAWN_WARNING', 
                                {'message': f'只能添加 {len(new_agents)}/{agents_to_add} 个智能体，因为空间限制'})
    
    def update(self, dt):
        """更新宇宙状态"""
        if self.perf_monitor:
            self.perf_monitor.start_update()
            
        self.frame_count += 1
        
        # 并行更新所有场
        self._update_fields_parallel(dt)

        # 更新空间网格
        self.update_spatial_grid()
        
        # 将智能体分成批次进行并行处理
        updated_agents = []
        agent_batches = [self.agents[i:i+BATCH_SIZE] for i in range(0, len(self.agents), BATCH_SIZE)]
        
        # 使用线程池并行处理智能体更新
        future_results = [self.thread_pool.submit(self._update_agent_batch, batch, dt) for batch in agent_batches]
        for future in future_results:
            updated_agents.extend(future.result())
        
        self.agents = updated_agents
        
        # 并行处理繁殖
        future_results = [self.thread_pool.submit(self._process_reproduction, batch) for batch in agent_batches]
        new_children = []
        for future in future_results:
            new_children.extend(future.result())
        
        # 添加新出生的智能体
        self.agents.extend(new_children)

        # 如果智能体数量低于最小阈值，补充新的随机智能体
        if len(self.agents) < MIN_AGENTS_TO_SPAWN:
            self._spawn_new_agents()
        
        # 如果智能体数量过多，淘汰一些能量最低的
        if len(self.agents) > MAX_AGENTS:
            self._cull_excess_agents()
            
        # 定期记录状态
        if self.frame_count % 20 == 0:
            self.logger.log_state(self.frame_count, self.agents)
            # 同时记录场景数据
            self.logger.log_field(self.frame_count, self.fields)
            # 记录信号类型
            self.logger.log_signal_types(self.signal_types)
            
        if self.perf_monitor:
            self.perf_monitor.end_update()
    
    def _cull_excess_agents(self):
        """淘汰多余的智能体"""
        self.agents.sort(key=lambda a: a.energy)
        num_to_remove = len(self.agents) - MAX_AGENTS
        culled_ids = [a.id for a in self.agents[:num_to_remove]]
        
        for agent_to_remove in self.agents[:num_to_remove]:
            agent_to_remove.is_dead = True
            self.on_agent_death(agent_to_remove)
            
        self.agents = self.agents[num_to_remove:]
        self.logger.log_event(self.frame_count, 'CULL', 
                             {'count': num_to_remove, 'culled_ids': culled_ids})
    
    def draw(self, surface, sim_surface):
        """绘制整个宇宙 - 使用批量渲染优化"""
        # 在无GUI模式下跳过渲染
        if not self.use_gui:
            return
            
        if self.perf_monitor:
            self.perf_monitor.start_render()
            
        # 背景
        sim_surface.fill((10, 10, 20))
        
        # 根据视图模式绘制场
        if self.view_mode == 0:
            # 显示所有场
            for field in self.fields:
                field.draw(sim_surface, self.camera)
        elif 1 <= self.view_mode <= len(self.fields):
            # 显示特定场
            self.fields[self.view_mode - 1].draw(sim_surface, self.camera, alpha=255)
            
        # 批量绘制所有智能体 - 按距离排序以确保正确的绘制顺序
        # 首先过滤掉不在视口内的智能体
        visible_agents = []
        for agent in self.agents:
            if agent.is_dead:
                continue
                
            # 检查是否在视口内
            screen_pos = self.camera.world_to_screen(agent.position)
            if (screen_pos[0] < -50 or screen_pos[0] > self.camera.render_width + 50 or
                screen_pos[1] < -50 or screen_pos[1] > self.camera.render_height + 50):
                continue
                
            visible_agents.append(agent)
        
        # 按基因型分组智能体，以便批量渲染
        if RENDER_OPTIMIZATION and USE_SURFACE_CACHING:
            # 创建基因型到智能体列表的映射
            genotype_groups = {}
            for agent in visible_agents:
                if agent is self.selected_agent:
                    # 选中的智能体单独绘制
                    agent.draw(sim_surface, self.camera)
                    continue
                    
                genotype_id = agent.genotype_id
                if genotype_id not in genotype_groups:
                    genotype_groups[genotype_id] = []
                genotype_groups[genotype_id].append(agent)
            
            # 为每个基因型批量绘制智能体
            for genotype_id, agents in genotype_groups.items():
                # 获取该基因型的颜色
                hue = (genotype_id * 20) % 360
                color = pygame.Color(0)
                color.hsva = (hue, 85, 90, 100)
                
                # 批量绘制相同基因型的智能体
                for i in range(0, len(agents), AGENT_RENDER_BATCH_SIZE):
                    batch = agents[i:i+AGENT_RENDER_BATCH_SIZE]
                    self._draw_agent_batch(sim_surface, batch, color)
        else:
            # 传统方式：逐个绘制
            for agent in visible_agents:
                agent.draw(sim_surface, self.camera)
            
        # 将模拟表面绘制到主表面
        surface.blit(sim_surface, (0, 0))
        
        if self.perf_monitor:
            self.perf_monitor.end_render()
    
    def _draw_agent_batch(self, surface, agents, color):
        """批量绘制一组相同基因型的智能体"""
        for agent in agents:
            # 计算屏幕位置
            screen_pos = self.camera.world_to_screen(agent.position)
            
            # 计算半径
            radius = max(1, int(agent.radius * self.camera.zoom))
            
            # 根据能量水平调整亮度
            energy_ratio = min(1.0, agent.energy / max(0.1, agent.e_repro))
            agent_color = color
            
            if energy_ratio < 0.3:
                # 能量不足时颜色变暗
                hue, s, v, a = color.hsva
                dark_color = pygame.Color(0)
                dark_color.hsva = (hue, s, max(30, int(v * energy_ratio / 0.3)), a)
                agent_color = dark_color
            
            # 绘制智能体
            if radius <= 2:
                pygame.draw.circle(surface, agent_color, screen_pos, 1)
            else:
                pygame.draw.circle(surface, agent_color, screen_pos, radius, 1)

    def handle_click(self, mouse_pos):
        """处理鼠标点击事件"""
        world_pos = self.camera.screen_to_world(mouse_pos)
        closest_agent = None
        min_dist_sq = (10 / self.camera.zoom)**2
        
        # 查找最近的智能体
        for agent in self.agents:
            dist_sq = (agent.position - world_pos).length_squared()
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_agent = agent
                
        self.selected_agent = closest_agent

    def _load_agents_from_log(self):
        """从日志加载智能体状态"""
        agents_data = self.logger.load_last_state()
        signal_types = self.logger.load_signal_types()
        self.signal_types = signal_types
        
        print(f"从日志加载 {len(agents_data)} 个智能体")
        
        for agent_row in agents_data:
            try:
                # 解析智能体数据
                agent_id = int(agent_row[1])
                parent_id = int(agent_row[2]) if agent_row[2] != "None" else None
                genotype_id = int(agent_row[3])
                is_mutant = agent_row[4].lower() == "true"
                energy = float(agent_row[5])
                pos_x = float(agent_row[6])
                pos_y = float(agent_row[7])
                gene_str = agent_row[11]
                
                # 解析基因
                gene = json.loads(gene_str.replace("'", "\""))
                
                # 创建智能体
                agent = Agent(
                    universe=self,
                    logger=self.logger,
                    gene=gene,
                    position=Vector2(pos_x, pos_y),
                    energy=energy,
                    parent_id=parent_id,
                    is_mutant=is_mutant
                )
                
                # 设置智能体ID和基因型ID
                agent.id = agent_id
                agent.genotype_id = genotype_id
                
                # 注册基因型
                canonical_gene = self._get_canonical_gene(gene)
                self.genotype_registry[canonical_gene] = genotype_id
                self.next_genotype_id = max(self.next_genotype_id, genotype_id + 1)
                
                # 添加到智能体列表
                self.agents.append(agent)
            except Exception as e:
                print(f"加载智能体时出错: {str(e)}")
    
    def _create_initial_agents(self):
        """创建初始智能体 - 确保位置不重叠"""
        occupied_positions = []
        
        # 创建指定数量的初始智能体
        for _ in range(INITIAL_AGENT_COUNT):
            valid_position = False
            max_attempts = 50  # 每个智能体尝试位置的最大次数
            
            for _ in range(max_attempts):
                # 生成随机位置
                candidate_pos = Vector2(
                    random.uniform(0, WORLD_SIZE),
                    random.uniform(0, WORLD_SIZE)
                )
                
                # 检查是否与现有智能体重叠
                valid_position = True
                
                for existing_pos in occupied_positions:
                    # 考虑周期性边界条件计算距离
                    dx = min(abs(candidate_pos.x - existing_pos.x), WORLD_SIZE - abs(candidate_pos.x - existing_pos.x))
                    dy = min(abs(candidate_pos.y - existing_pos.y), WORLD_SIZE - abs(candidate_pos.y - existing_pos.y))
                    dist_sq = dx * dx + dy * dy
                    
                    if dist_sq < AGENT_RADIUS * 3.0 * AGENT_RADIUS * 3.0:
                        valid_position = False
                        break
                
                # 如果找到有效位置，创建智能体
                if valid_position:
                    # 创建新智能体并添加到列表
                    agent = Agent(self, self.logger, position=candidate_pos)
                    self.agents.append(agent)
                    occupied_positions.append(candidate_pos)
                    break
            
            # 如果无法找到有效位置，记录警告
            if not valid_position:
                self.logger.log_event(0, 'SPAWN_WARNING', 
                                    {'message': f'无法为智能体 #{_+1} 找到合适位置'})
        
        # 记录实际创建的智能体数量
        actual_count = len(self.agents)
        if actual_count < INITIAL_AGENT_COUNT:
            self.logger.log_event(0, 'SPAWN_WARNING', 
                                {'message': f'Only created {actual_count}/{INITIAL_AGENT_COUNT} agents due to space constraints'})

# --- UI组件 ---
def draw_inspector_panel(surface, font, agent, mouse_pos, panel_x, panel_width, panel_height):
    """绘制智能体观察面板"""
    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_surface.fill((40, 40, 60, 220))
    surface.blit(panel_surface, (panel_x, 0))
    
    # 如果没有选中智能体，显示提示信息
    if agent is None:
        text = font.render("点击一个生命体来观察", True, (200, 200, 200))
        surface.blit(text, (panel_x + 20, 20))
        return
    
    # 绘制智能体信息
    y_offset = 20
    def draw_text(text, value, color=(255, 255, 255)):
        nonlocal y_offset
        text_surf = font.render(f"{text}: {value}", True, color)
        surface.blit(text_surf, (panel_x + 20, y_offset))
        y_offset += 25
    
    # 基本信息（突变体特殊标记）
    if agent.is_mutant:
        draw_text("观察对象 ID", f"{agent.id} (M)", (255, 255, 100))
    else:
        draw_text("观察对象 ID", agent.id, (100, 255, 100))

    draw_text("亲代 ID", agent.parent_id if agent.parent_id else "N/A")
    draw_text("基因型 ID", agent.genotype_id)
    draw_text("能量 (E)", f"{agent.energy:.2f}")
    draw_text("位置 (p)", f"({agent.position.x:.1f}, {agent.position.y:.1f})")
    
    # 基因特性
    y_offset += 10
    draw_text("--- 基因特性 ---", "", (200, 200, 100))
    draw_text("复杂度 (Ω)", agent.complexity)
    
    # 获取基础节点信息
    base_input = agent.gene.get('base_input', 12)
    base_output = agent.gene.get('base_output', 2)
    
    # 显示节点数量信息，标记出额外节点
    extra_input = agent.n_input - base_input
    extra_output = agent.n_output - base_output
    input_text = f"{agent.n_input}"
    if extra_input > 0:
        input_text += f" (基础{base_input} + 额外{extra_input})"
    draw_text("输入节点数", input_text)
    
    output_text = f"{agent.n_output}"
    if extra_output > 0:
        output_text += f" (移动{base_output} + 信号{extra_output})"
    draw_text("输出节点数", output_text)
    
    draw_text("隐藏节点数", agent.n_hidden)
    draw_text("连接数", len(agent.gene['connections']))
    draw_text("思维深度 (k)", agent.computation_depth)
    draw_text("环境吸收系数", f"{agent.env_absorption_coeff:.2f}")
    
    # 生态特性
    y_offset += 10
    draw_text("--- 生态特性 ---", "", (200, 100, 200))
    
    # 只显示身份向量
    id_value = round(agent.identity_vector, 2)
    draw_text("身份向量", f"{id_value:.2f}", 
             (int(100 + abs(id_value) * 100), 
              int(100 + (1 - abs(id_value)) * 100), 
              int(200 - abs(id_value) * 100)))
    
    # 行为输出
    y_offset += 10
    draw_text("--- 行为输出 ---", "", (200, 200, 100))
    
    # 确保即使在节点数量变化的情况下也能正确显示
    if len(agent.last_action_vector) > 0:
        draw_text("移动 X", f"{agent.last_action_vector[0]:.2f}")
    if len(agent.last_action_vector) > 1:
        draw_text("移动 Y", f"{agent.last_action_vector[1]:.2f}")
    
    # 显示所有信号输出
    if len(agent.last_action_vector) > 2:
        for i, signal in enumerate(agent.last_action_vector[2:], 1):
            draw_text(f"信号{i}强度", f"{abs(signal):.2f}")
    
    # 神经网络可视化
    y_offset += 20
    draw_neural_network(surface, font, agent, panel_x + 20, y_offset, panel_width - 40, 350, mouse_pos)

def draw_neural_network(surface, font, agent, x, y, width, height, mouse_pos):
    """绘制神经网络可视化"""
    title = font.render("计算核心 (Cᵢ) 拓扑图:", True, (200, 200, 100))
    surface.blit(title, (x, y))
    y += 30
    
    # 获取节点数量
    n_in, n_hid, n_out = agent.n_input, agent.n_hidden, agent.n_output
    
    # 从基因中获取节点类型信息
    node_types = agent.gene.get('node_types', {
        'input': ['env_sense'] * n_in,
        'output': ['movement'] * 2 + ['signal'] * (n_out - 2) if n_out > 2 else ['movement'] * n_out,
        'hidden': ['standard'] * n_hid
    })
    
    # 为所有节点类型创建标签
    input_labels = []
    # 使用统一的signal命名方式
    basic_input_labels = ["Energy_v", "Energy_gx", "Energy_gy", "Hazard_v", "Hazard_gx", "Hazard_gy", 
                         "Signal1_v", "Signal1_gx", "Signal1_gy", "Signal2_v", "Signal2_gx", "Signal2_gy"]
    signal_in_count = 0
    for i in range(n_in):
        node_type = None
        if 'node_types' in agent.gene and 'input' in agent.gene['node_types'] and i < len(agent.gene['node_types']['input']):
            node_type = agent.gene['node_types']['input'][i]
        if node_type == 'signal_sense':
            input_labels.append(f"Signal{signal_in_count+3}_v")
            signal_in_count += 1
        elif i < len(basic_input_labels):
            input_labels.append(basic_input_labels[i])
        else:
            input_labels.append(f"In_{i}")
    
    # 输出标签
    output_labels = []
    # 基础移动输出
    if n_out > 0:
        output_labels.append("MoveX")
    if n_out > 1:
        output_labels.append("MoveY")
    # 信号输出
    for i in range(2, n_out):
        output_labels.append(f"Signal_{i-1}")
    
    # 设置列位置
    col_x = [x + 30, x + width // 2, x + width - 30]
    layers = [n_in, n_hid, n_out]
    node_positions = {}
    
    # 根据是否有隐藏层决定布局
    col_map = [0, 1, 2] if n_hid > 0 else [0, 2] 
    
    # 计算每个节点的位置
    current_node_idx = 0
    visible_layer_idx = 0 
    for i, n_nodes in enumerate(layers):
        if n_nodes == 0: 
            continue
        
        # 计算当前层的起始Y位置（使节点分布均匀）
        layer_y_start = y + (height - (n_nodes - 1) * 25) / 2 if n_nodes > 1 else y + height / 2
        
        # 为每个节点分配位置
        for j in range(n_nodes):
            node_id = current_node_idx + j
            column_to_use = col_x[col_map[visible_layer_idx]]
            node_positions[node_id] = (int(column_to_use), int(layer_y_start + j * 25))

        current_node_idx += n_nodes
        visible_layer_idx += 1

    # 绘制连接
    for from_n, to_n, weight in agent.gene['connections']:
        if from_n in node_positions and to_n in node_positions:
            start_pos, end_pos = node_positions[from_n], node_positions[to_n]
            # 使用颜色表示权重符号（绿色为正，红色为负）
            line_width = min(3, max(1, abs(int(weight * 2))))
            color = (0, min(255, 100 + int(abs(weight) * 80)), 0) if weight > 0 else (min(255, 150 + int(abs(weight) * 50)), 50, 50)
            pygame.draw.line(surface, color, start_pos, end_pos, line_width)
    
    # 处理鼠标悬停信息
    hover_info = None
    
    # 绘制节点
    for node_id, pos in node_positions.items():
        is_input = node_id < n_in
        is_hidden = n_in <= node_id < n_in + n_hid
        is_output = node_id >= n_in + n_hid
        
        # 根据节点类型设置颜色
        if is_input:
            # 输入节点：环境感知为蓝色，额外输入为紫色
            if node_id < len(basic_input_labels):
                color = (100, 100, 255)  # 蓝色: 环境感知
            else:
                color = (180, 100, 255)  # 紫色: 额外感知
        elif is_hidden:
            color = (255, 165, 0)  # 橙色: 隐藏节点
        else:
            # 输出节点：移动为黄色，信号为绿色
            output_idx = node_id - (n_in + n_hid)
            if output_idx < 2:  # 移动节点
                color = (255, 255, 100)  # 黄色
            else:  # 信号节点
                color = (100, 255, 100)  # 绿色
        
        # 根据激活值调整颜色亮度
        activation = agent.node_activations[node_id]
        brightness = max(0, min(255, 128 + int(activation * 127)))
        color = tuple(min(255, c * brightness // 128) for c in color)
        
        # 绘制节点圆圈
        radius = 6
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, (0,0,0), pos, radius, 1)

        # 设置标签
        label = None
        if is_input and node_id < len(input_labels):
            label = input_labels[node_id]
        elif is_output:
            output_idx = node_id - (n_in + n_hid)
            if output_idx < len(output_labels):
                label = output_labels[output_idx]

        # 绘制标签
        if label:
            label_surf = font.render(label, True, (200, 200, 200))
            if is_input:
                surface.blit(label_surf, (pos[0] - label_surf.get_width() - 5, pos[1] - 8))
            else:
                surface.blit(label_surf, (pos[0] + 10, pos[1] - 8))
        
        # 检测鼠标悬停
        if math.hypot(mouse_pos[0] - pos[0], mouse_pos[1] - pos[1]) < radius:
            # 根据节点类型创建不同的悬停信息
            if is_input:
                node_type = "环境感知" if node_id < len(basic_input_labels) else "额外输入"
            elif is_hidden:
                node_type = "隐藏"
            else:
                output_idx = node_id - (n_in + n_hid)
                if output_idx < 2:
                    node_type = "移动" 
                else:
                    node_type = "信号"
            
            hover_info = (f"{node_type}节点 {node_id}", f"激活值: {agent.node_activations[node_id]:.3f}", mouse_pos)

    # 绘制悬停信息
    if hover_info:
        title, value, pos = hover_info
        title_surf = font.render(title, True, (255, 255, 255))
        value_surf = font.render(value, True, (255, 255, 255))
        box_rect = pygame.Rect(pos[0] + 10, pos[1] + 10, 
                              max(title_surf.get_width(), value_surf.get_width()) + 20, 50)
        pygame.draw.rect(surface, (0,0,0,200), box_rect)
        surface.blit(title_surf, (box_rect.x + 10, box_rect.y + 5))
        surface.blit(value_surf, (box_rect.x + 10, box_rect.y + 25))

# --- 性能监控系统 ---
class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.update_times = []
        self.render_times = []
        self.agent_counts = []
        self.last_time = time.time()
        self.last_stats_time = time.time()
        self.fps = 0
        self.avg_update_time = 0
        self.avg_render_time = 0
        self.cpu_usage = 0
        
    def start_frame(self):
        self.last_time = time.time()
        
    def end_frame(self, agent_count):
        current_time = time.time()
        frame_time = current_time - self.last_time
        
        self.frame_times.append(frame_time)
        self.agent_counts.append(agent_count)
        
        # 保持最近100帧的数据
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
            self.agent_counts.pop(0)
        
        # 计算FPS
        if self.frame_times and sum(self.frame_times) > 0:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        else:
            self.fps = 0
    
    def start_update(self):
        self.update_start_time = time.time()
    
    def end_update(self):
        update_time = time.time() - self.update_start_time
        self.update_times.append(update_time)
        if len(self.update_times) > 100:
            self.update_times.pop(0)
        if self.update_times:
            self.avg_update_time = sum(self.update_times) / len(self.update_times)
        else:
            self.avg_update_time = 0
    
    def start_render(self):
        self.render_start_time = time.time()
    
    def end_render(self):
        render_time = time.time() - self.render_start_time
        self.render_times.append(render_time)
        if len(self.render_times) > 100:
            self.render_times.pop(0)
        if self.render_times:
            self.avg_render_time = sum(self.render_times) / len(self.render_times)
        else:
            self.avg_render_time = 0
    
    def get_stats(self):
        return {
            'fps': round(self.fps, 1),
            'agents': self.agent_counts[-1] if self.agent_counts else 0,
            'update_ms': round(self.avg_update_time * 1000, 1),
            'render_ms': round(self.avg_render_time * 1000, 1)
        }

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="涌现认知生态系统 (ECE) v5.0")
    parser.add_argument("--no-gui", action="store_true", help="无GUI模式，仅运行计算")
    parser.add_argument("--continue-from", type=str, help="从指定的日志目录继续模拟，新日志将保存在新的日志目录中")
    args = parser.parse_args()
    
    # 使用GUI模式
    use_gui = not args.no_gui
    
    # 初始化数据记录器
    if args.continue_from:
        logger = DataLogger(args.continue_from)
        continue_simulation = True
    else:
        logger = DataLogger()
        continue_simulation = False
    
    # 如果使用GUI模式，初始化pygame
    if use_gui:
        # 设置窗口位置居中
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        
        # 设置Pygame以提高性能
        pygame.init()
        pygame.display.set_caption("涌现认知生态系统 (ECE) v5.0 - 高性能版")
        
        # 设置显示模式
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT), flags)
        
        # 使用硬件加速
        if pygame.display.get_driver() == 'windows':
            # 在Windows上尝试使用DirectX
            os.environ['SDL_VIDEODRIVER'] = 'directx'
        
        # 移除帧率限制
        clock = pygame.time.Clock()
        
        # 设置字体
        try: 
            font = pygame.font.SysFont("simhei", 16)
        except pygame.error: 
            font = pygame.font.SysFont(None, 22)
        
        # 设置模拟区域大小
        current_screen_width, current_screen_height = screen.get_size()
        sim_area_width = current_screen_width - INFO_PANEL_WIDTH
    else:
        # 无GUI模式
        print("以无GUI模式运行，仅进行计算...")
        sim_area_width = INITIAL_SCREEN_WIDTH - INFO_PANEL_WIDTH
        current_screen_height = INITIAL_SCREEN_HEIGHT
    
    # 创建宇宙
    universe = Universe(logger, sim_area_width, current_screen_height, use_gui, continue_simulation)
    
    # 记录模拟开始事件
    if not continue_simulation:
        logger.log_event(0, 'SIM_START', {'initial_agents': INITIAL_AGENT_COUNT, 'world_size': WORLD_SIZE, 'gui_mode': use_gui})
    else:
        logger.log_event(universe.frame_count, 'SIM_CONTINUE', {'agents': len(universe.agents), 'from_frame': universe.frame_count})
    
    # 控制变量
    running = True
    paused = False
    last_performance_update = 0
    
    # 渲染优化变量
    render_every_n_frames = DEFAULT_RENDER_SKIP  # 使用默认配置
    frame_counter = 0
    
    # 主循环
    while running:
        frame_counter += 1
        render_this_frame = frame_counter % render_every_n_frames == 0
        
        if universe.perf_monitor and render_this_frame:
            universe.perf_monitor.start_frame()
            
        # GUI模式下处理事件
        if use_gui:
            mouse_pos = pygame.mouse.get_pos()
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    running = False
                
                # 窗口大小调整
                if event.type == pygame.VIDEORESIZE:
                    current_screen_width, current_screen_height = event.size
                
                # 相机事件处理
                universe.camera.handle_event(event, mouse_pos)
                
                # 鼠标点击
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and mouse_pos[0] < universe.camera.render_width:
                        universe.handle_click(event.pos)
                        
                # 键盘控制
                if event.type == pygame.KEYDOWN:
                    # 空格暂停/继续
                    if event.key == pygame.K_SPACE: 
                        paused = not paused
                        
                    # 右箭头在暂停时单步执行
                    if event.key == pygame.K_RIGHT and paused: 
                        universe.update(0.016)
                        
                    # F11全屏切换
                    if event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                        
                    # 数字键切换视图模式
                    if event.key == pygame.K_0:
                        universe.view_mode = 0
                    elif event.key == pygame.K_1:
                        universe.view_mode = 1
                    elif event.key == pygame.K_2:
                        universe.view_mode = 2
                    elif event.key == pygame.K_3:
                        universe.view_mode = 3
                    elif event.key == pygame.K_4:
                        universe.view_mode = 4
                        
                    # 优化键 - 调整渲染频率
                    elif event.key == pygame.K_F1:
                        render_every_n_frames = 1  # 每帧渲染
                    elif event.key == pygame.K_F2:
                        render_every_n_frames = 2  # 每2帧渲染
                    elif event.key == pygame.K_F3:
                        render_every_n_frames = 3  # 每3帧渲染
        else:
            # 无GUI模式下简单处理中断信号
            try:
                # 每100帧输出一次状态信息
                if universe.frame_count % 100 == 0:
                    total_biomass = sum(agent.energy for agent in universe.agents)
                    print(f"帧: {universe.frame_count} | 生命体: {len(universe.agents)}/{MAX_AGENTS} | 总生物量: {int(total_biomass)}")
            except KeyboardInterrupt:
                print("收到中断信号，正在退出...")
                running = False
        
        # 非暂停状态下更新模拟
        if not paused:
            # 使用固定的时间步长，提高模拟稳定性
            fixed_dt = 0.016  # 约60FPS
            universe.update(fixed_dt)
        
        # 只在需要渲染的帧上执行渲染，且仅在GUI模式下
        if use_gui and render_this_frame:
            # 清除屏幕
            screen.fill((0,0,0))
            
            # 更新屏幕布局
            current_screen_width, current_screen_height = screen.get_size()
            sim_area_width = current_screen_width - INFO_PANEL_WIDTH
            if sim_area_width < 400: 
                sim_area_width = 400
            info_panel_width = current_screen_width - sim_area_width
            universe.camera.update_render_size(sim_area_width, current_screen_height)
            
            # 创建模拟表面
            sim_surface = pygame.Surface((sim_area_width, current_screen_height))
            
            # 绘制宇宙和信息面板
            universe.draw(screen, sim_surface)
            draw_inspector_panel(screen, font, universe.selected_agent, mouse_pos, 
                                sim_area_width, info_panel_width, current_screen_height)
            
            # 显示当前视图模式
            view_name = "全部"
            if 1 <= universe.view_mode <= len(universe.fields):
                view_name = universe.fields[universe.view_mode - 1].name
            
            # 显示状态信息
            total_biomass = sum(agent.energy for agent in universe.agents)
            
            # 性能统计
            performance_info = ""
            if universe.perf_monitor and universe.frame_count - last_performance_update > UPDATE_INTERVAL:
                stats = universe.perf_monitor.get_stats()
                performance_info = f" | FPS: {stats['fps']} | 更新: {stats['update_ms']}ms | 渲染: {stats['render_ms']}ms"
                last_performance_update = universe.frame_count
            
            # 显示完整状态文本 - 恢复原有内容
            info_text = f"帧: {universe.frame_count} | 生命体: {len(universe.agents)}/{MAX_AGENTS} ({universe.next_genotype_id}个基因型) | " \
                       f"总生物量: {int(total_biomass)} | 视图(0-4): {view_name}{performance_info} | {'[已暂停]' if paused else ''}"
            text_surface = font.render(info_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
            
            # 显示渲染频率
            if render_every_n_frames > 1:
                render_text = f"渲染频率: 每{render_every_n_frames}帧 (F1-F3调整)"
                render_surface = font.render(render_text, True, (255, 200, 100))
                screen.blit(render_surface, (10, 30))
            
            # 更新屏幕
            pygame.display.flip()
            
            if universe.perf_monitor:
                universe.perf_monitor.end_frame(len(universe.agents))
            
            # GUI模式下控制帧率
            # clock.tick(60)  # 不限制帧率
    
    # 确保退出前刷新日志缓冲区
    logger._flush_buffers()
    
    # 如果使用了pygame，则退出
    if use_gui:
        pygame.quit()
    print("模拟结束")

if __name__ == '__main__':
    main() 