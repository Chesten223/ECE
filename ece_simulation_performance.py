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
import multiprocessing
from multiprocessing import Queue
import pygame
import numpy as np
import random
import math
from numba import njit
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
WORLD_SIZE = 2048
INFO_PANEL_WIDTH = 400

# 2. 演化引擎参数
INITIAL_AGENT_COUNT = 100  # 初始智能体数量设置为100
MAX_AGENTS = 1000          # 限制最大智能体数量为500
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
MOVEMENT_ENERGY_COST = 0.0004   # 略微减少移动能耗
SIGNAL_EMISSION_RADIUS = 20.0 
BIOTIC_FIELD_SPECIAL_DECAY = 2.0
AGENT_RADIUS = 2.0
# COLLISION_STIFFNESS = 5.0      # 【新增】碰撞“硬度”，值越大，推力越强，生物越“硬” /////那个过于暴力的 COLLISION_STIFFNESS 参数是问题的根源，我们不再需要它了。 （好惨啊
MILD_REPULSION_RADIUS = 0   # 排斥力作用范围
MILD_REPULSION_STRENGTH = 0  # 排斥力强度
COLLISION_ITERATIONS = 5       # 碰撞检测迭代次数
HIGH_DENSITY_THRESHOLD = 3     # 高密度区域的邻居数量阈值
OVERLAP_EMERGENCY_DISTANCE = 0.5  # 紧急情况下的额外排斥距离
MIN_MOVEMENT_JITTER = 0     # 最小随机移动量，确保所有生物都会动
REPULSION_PRIORITY = 0       # 排斥力优先级，确保排斥力优先于神经网络输出
ENERGY_PATCH_RADIUS_MIN = 60.0 # 能量辐射最小范围 (原来是30)
ENERGY_PATCH_RADIUS_MAX = 400.0 # 能量辐射最大范围 (原来是60)
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


def logging_process_worker(log_queue, log_dir, continue_from, headers):
    """
    这个函数在独立的进程中运行，专门处理日志写入。
    它从队列中接收数据，执行耗时的编码和磁盘写入操作。
    """
    # 在新进程中定义文件路径
    state_log_path = os.path.join(log_dir, "simulation_log.csv")
    event_log_path = os.path.join(log_dir, "event_log.csv")
    field_log_path = os.path.join(log_dir, "field_log.csv")
    signal_types_path = os.path.join(log_dir, "signal_types.json")

    # 如果不是继续模拟，则创建新文件并写入表头
    if not continue_from:
        with open(state_log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers['state'])
        with open(event_log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers['event'])
        with open(field_log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers['field'])

    # 持续从队列中获取数据并写入
    while True:
        try:
            log_item = log_queue.get()
            if log_item is None:  # 收到结束信号
                break

            log_type, data = log_item
            if log_type == 'state':
                with open(state_log_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(data)
            elif log_type == 'event':
                 with open(event_log_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(data)
            elif log_type == 'field':
                frame, field_name, grid_data = data
                # 在这个进程里进行耗时的编码操作
                field_bytes = grid_data.tobytes()
                encoded_data = base64.b64encode(field_bytes).decode('ascii')
                with open(field_log_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([frame, field_name, encoded_data])
            elif log_type == 'signal_types':
                with open(signal_types_path, 'w', encoding='utf-8') as f:
                    json.dump(list(data), f)

        except Exception as e:
            # 在日志进程中打印错误，避免主进程崩溃
            print(f"[Logging Process Error]: {e}")


# --- 数据日志系统 ---
# --- 数据日志系统 ---
class DataLogger:
    def __init__(self, log_queue, continue_from=None):
        self.log_queue = log_queue
        self.continue_from = continue_from
        
        if continue_from:
            self.agent_id_counter = self._get_max_agent_id(continue_from)
            self.last_frame = self._get_last_frame(continue_from)
        else:
            self.agent_id_counter = 0
            self.last_frame = 0

        # 缓冲区，用于批量提交到队列
        self.state_buffer = []
        self.event_buffer = []
        self.buffer_size_limit = LOG_BUFFER_SIZE
        self.last_flush_time = time.time()
        self.flush_interval = LOG_FLUSH_INTERVAL

    def get_new_agent_id(self):
        self.agent_id_counter += 1
        return self.agent_id_counter

    def log_state(self, frame_number, agents):
        for agent in agents:
            gene_str = str(agent.gene)
            row = [frame_number, agent.id, agent.parent_id, agent.genotype_id, agent.is_mutant, 
                   round(agent.energy, 2), round(agent.position.x, 2), round(agent.position.y, 2), 
                   agent.gene['n_hidden'], len(agent.gene['connections']), gene_str]
            self.state_buffer.append(row)
        self._check_flush_buffer()

    def log_event(self, frame, event_type, details):
        details_str = json.dumps(details)
        self.event_buffer.append([frame, event_type, details_str])
        self._check_flush_buffer()
    
    def log_field(self, frame, fields):
        # 只把原始的、廉价的数据放入队列
        # 耗时的base64编码将由日志进程完成
        for field in fields:
            grid_data = np.array(field.grid, dtype=np.float32)
            self.log_queue.put(('field', (frame, field.name, grid_data)))
            
    def log_signal_types(self, signal_types):
        self.log_queue.put(('signal_types', signal_types))

    def _check_flush_buffer(self):
        current_time = time.time()
        buffer_full = (len(self.state_buffer) + len(self.event_buffer)) >= self.buffer_size_limit
        time_to_flush = current_time - self.last_flush_time > self.flush_interval
        
        if buffer_full or time_to_flush:
            self._flush_buffers()

    def _flush_buffers(self):
        if self.state_buffer:
            self.log_queue.put(('state', self.state_buffer))
            self.state_buffer = []
        if self.event_buffer:
            self.log_queue.put(('event', self.event_buffer))
            self.event_buffer = []
        self.last_flush_time = time.time()

    def _get_max_agent_id(self, log_dir):
        # ... (这个函数和后面的 _get_last_frame, load_last_state 等保持不变，用于初始化) ...
        max_id = 0
        try:
            path = os.path.join(log_dir, "simulation_log.csv")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) > 1: max_id = max(max_id, int(row[1]))
        except Exception: pass
        return max_id

    def _get_last_frame(self, log_dir):
        last_frame = 0
        try:
            path = os.path.join(log_dir, "simulation_log.csv")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) > 0: last_frame = max(last_frame, int(row[0]))
        except Exception: pass
        return last_frame

    def load_last_state(self):
        agents_data = []
        try:
            path = os.path.join(self.continue_from, "simulation_log.csv")
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) > 0 and int(row[0]) == self.last_frame:
                        agents_data.append(row)
        except Exception: pass
        return agents_data

    def load_signal_types(self):
        signal_types = set()
        try:
            path = os.path.join(self.continue_from, "signal_types.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    signal_types = set(json.load(f))
        except Exception: pass
        return signal_types

    def __del__(self):
        # 确保退出时刷新所有剩余的缓冲数据
        self._flush_buffers()

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
        n_input = random.randint(2, 5)
        n_output = random.randint(1, 3)
        n_hidden = random.randint(0, 2)
        
        # 创建极少数的随机连接
        connections = []
        
        # 确保至少有1个连接
        if n_input > 0 and (n_hidden + n_output) > 0:
            from_node = random.randint(0, n_input - 1)
            to_node = random.randint(n_input, n_input + n_hidden + n_output - 1)
            connections.append([from_node, to_node, random.uniform(-1, 1)])
        
        # 有50%概率添加第二个连接
        if random.random() < 0.5 and n_input + n_hidden > 0 and n_hidden + n_output > 0:
            from_node = random.randint(0, n_input + n_hidden - 1)
            to_node = random.randint(n_input, n_input + n_hidden + n_output - 1)
            if to_node > from_node:
                connections.append([from_node, to_node, random.uniform(-1, 1)])
        
        # 极简的节点类型初始化
        input_types = []
        for _ in range(n_input):
            input_types.append(random.choice(['field_sense', 'signal_sense']))
            
        output_types = []
        for _ in range(n_output):
            output_types.append(random.choice(['move_vector', 'signal']))
            
        hidden_types = []
        for _ in range(n_hidden):
            hidden_types.append('standard')
        
        # 返回极简的基因结构
        return {
            'n_input': n_input,
            'n_output': n_output, 
            'n_hidden': n_hidden, 
            # 【修改】删除 computation_depth
            'connections': connections,
            'env_absorption_coeff': random.uniform(-0.5, 0.5),
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
        
        # 【修改】 computation_depth 不再需要，延迟由结构决定
        # self.computation_depth = self.gene['computation_depth'] 
        
        # 计算总节点数
        total_nodes = self.n_input + self.n_hidden + self.n_output
        
        # 【修改】初始化两个状态数组
        self.node_activations = np.zeros(total_nodes, dtype=np.float32)
        self.next_node_inputs = np.zeros(total_nodes, dtype=np.float32)
        
        # 初始化连接矩阵 - 保持不变
        self.connection_matrix = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        
        for from_node, to_node, weight in self.gene['connections']:
            if 0 <= from_node < total_nodes and 0 <= to_node < total_nodes:
                self.connection_matrix[from_node, to_node] = weight
        
        self.output_start_idx = self.n_input + self.n_hidden
        self.last_action_vector = np.zeros(self.n_output, dtype=np.float32)
        
        # 【修改】复杂度计算中移除 computation_depth
        self.complexity = self.n_hidden + len(self.gene['connections']) * 0.5
        
        # 设置繁殖参数
        self.e_repro = 20 + self.complexity * 5
        self.e_child = 10 + self.complexity * 2
        
        # 设置基础代谢成本
        self.metabolism_cost = 0.01 + self.complexity * 0.002
        
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
        
        # self.age += 1 # 如果您启用了年龄淘汰

        # AI决策阶段 (不变)
        perception_vector = self.universe.get_perception_vector(self.position)
        num_senses_to_use = min(len(perception_vector), self.n_input)
        self.node_activations[:num_senses_to_use] = perception_vector[:num_senses_to_use]
        self.next_node_inputs.fill(0)
        propagated_signals = np.matmul(self.node_activations, self.connection_matrix)
        self.next_node_inputs += propagated_signals
        hidden_output_inputs = self.next_node_inputs[self.n_input:]
        activated_values = np.tanh(hidden_output_inputs)
        self.node_activations[self.n_input:] = activated_values
        output_activations = self.node_activations[-self.n_output:]
        self.last_action_vector = output_activations
        
        # 行动意图生成阶段
        move_vector = Vector2(0, 0)
        move_components = []
        for i, activation in enumerate(output_activations):
            node_type = 'signal'
            if 'node_types' in self.gene and 'output' in self.gene['node_types'] and i < len(self.gene['node_types']['output']):
                node_type = self.gene['node_types']['output'][i]

            if node_type == 'move_vector':
                move_components.append(activation)
            elif node_type == 'signal':
                # ... (信号逻辑不变) ...
                signal_count = len(self.universe.fields) - 2
                field_idx = (i % signal_count) + 2
                if field_idx < len(self.universe.fields) and abs(activation) > SIGNAL_RENDER_THRESHOLD:
                    signal_strength = abs(activation) * 0.02
                    signal_radius = SIGNAL_EMISSION_RADIUS * (0.5 + abs(activation) * 0.5)
                    self.universe.fields[field_idx].add_circular_source(self.position, signal_radius, signal_strength)
                    self.universe.signal_types.add(f"Signal {field_idx-1}")
        
        # 【【【 恢复为简洁逻辑 】】】
        # 因为突变已保证 move_components 总是偶数个，所以不再需要处理落单情况
        for i in range(0, len(move_components), 2):
            if i + 1 < len(move_components):
                vx = move_components[i]
                vy = move_components[i+1]
                move_vector += Vector2(vx, vy)

        self.position += move_vector * MOVEMENT_SPEED_FACTOR * dt

        # 【生理活动阶段】
        self.position.x %= WORLD_SIZE
        self.position.y %= WORLD_SIZE

        for other in neighbors:
            if other is self or other.is_dead: continue
            dist_sq = (self.position - other.position).length_squared()
            if dist_sq < INTERACTION_RANGE**2:
                identity_diff = abs(self.identity_vector - other.identity_vector)
                OPTIMAL_DIFF = 0.5
                predation_efficiency = math.exp(-10 * (identity_diff - OPTIMAL_DIFF)**2)
                if self.identity_vector > other.identity_vector and predation_efficiency > 0.1:
                    dist_factor = 1 - math.sqrt(dist_sq) / INTERACTION_RANGE
                    energy_transfer = predation_efficiency * K_INTERACTION_FACTOR * 30 * dist_factor
                    self.energy += energy_transfer * dt
                    other.energy -= energy_transfer * dt
                    if energy_transfer * dt > 1.0 and random.random() < 0.05:
                        self.universe.logger.log_event(self.universe.frame_count, 'PREDATION', {'pred_id': self.id, 'prey_id': other.id})

        action_cost = move_vector.length_squared() * MOVEMENT_ENERGY_COST
        signal_cost = sum(abs(a) for a in output_activations) * 0.1
        metabolism = self.metabolism_cost + action_cost + signal_cost
        nutrient_val, _ = self.universe.nutrient_field.get_value_and_gradient(self.position)
        hazard_val, _ = self.universe.hazard_field.get_value_and_gradient(self.position)
        env_gain = self.env_absorption_coeff * nutrient_val * 40
        env_loss = abs(np.tanh(self.identity_vector)) * hazard_val * 30
        self.energy += (env_gain - env_loss - metabolism) * dt

        if self.energy <= 0:
            self.is_dead = True
            self.logger.log_event(self.universe.frame_count, 'AGENT_DEATH', {'agent_id': self.id, 'reason': 'energy_depleted'})
            self.universe.on_agent_death(self)

    def reproduce(self):
        # 繁殖检查：能量必须达到繁殖阈值
        if self.energy < self.e_repro:
            return None

        # 使用空间网格获取周围智能体
        neighbors = self.universe.get_neighbors(self)
        
        # 尝试找到一个没有重叠的位置
        max_attempts = 30
        child_pos = None
        min_safe_distance = self.radius * 2.5
        
        neighbor_positions = []
        for agent in neighbors:
            if agent is not self and not agent.is_dead:
                neighbor_positions.append(agent.position)
        
        for attempt in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            distance_factor = 1.0 + attempt * 0.1
            distance = random.uniform(self.radius * 2.0, self.radius * 10.0 * distance_factor)
            candidate_pos = Vector2(
                self.position.x + math.cos(angle) * distance,
                self.position.y + math.sin(angle) * distance
            )
            
            candidate_pos.x %= WORLD_SIZE
            candidate_pos.y %= WORLD_SIZE
            
            is_valid = True
            for pos in neighbor_positions:
                dx = min(abs(candidate_pos.x - pos.x), WORLD_SIZE - abs(candidate_pos.x - pos.x))
                dy = min(abs(candidate_pos.y - pos.y), WORLD_SIZE - abs(candidate_pos.y - pos.y))
                dist_sq = dx * dx + dy * dy
                
                if dist_sq < min_safe_distance * min_safe_distance:
                    is_valid = False
                    break
            
            if is_valid:
                child_pos = candidate_pos
                break
        
        if child_pos is None:
            self.logger.log_event(self.universe.frame_count, 'REPRODUCTION_FAILED', 
                                 {'agent_id': self.id, 'reason': 'no_valid_position', 
                                  'neighbors': len(neighbor_positions)})
            return None

        reproduction_cost = self.e_child
        extra_cost = self.e_child * 0.2
        total_cost = reproduction_cost + extra_cost
        
        self.energy -= total_cost
        child_energy = reproduction_cost
        
        new_gene = json.loads(json.dumps(self.gene))
        mutations_occurred = []

        # ===== 基因连接突变 =====
        for conn in new_gene['connections']:
            if random.random() < MUTATION_PROBABILITY['point']:
                conn[2] += random.uniform(-1, 1) * MUTATION_STRENGTH
                conn[2] = max(-4.0, min(4.0, conn[2])) 
                mutations_occurred.append('point_mutation')
                
        if random.random() < MUTATION_PROBABILITY['add_conn']:
            total_nodes = new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']
            if new_gene['n_input'] > 0 and total_nodes > new_gene['n_input']:
                from_n = random.randint(0, new_gene['n_input'] + new_gene['n_hidden'] - 1)
                to_n = random.randint(new_gene['n_input'], total_nodes - 1)
                if to_n > from_n:
                    new_gene['connections'].append([from_n, to_n, random.uniform(-1, 1)])
                    mutations_occurred.append('add_connection')
            
        if random.random() < MUTATION_PROBABILITY['del_conn'] and len(new_gene['connections']) > 0:
            new_gene['connections'].pop(random.randrange(len(new_gene['connections'])))
            mutations_occurred.append('delete_connection')
            
        # ===== 神经网络参数突变 =====
        if 'env_absorption_coeff' in new_gene and random.random() < MUTATION_PROBABILITY['point']:
            new_gene['env_absorption_coeff'] += random.uniform(-1, 1) * MUTATION_STRENGTH
            mutations_occurred.append('absorption_coeff_mutation')
            
        # ===== 节点突变 =====
        if random.random() < MUTATION_PROBABILITY['add_node'] * 0.5:
            new_gene['n_input'] += 1
            if 'node_types' in new_gene:
                new_type = random.choice(['field_sense', 'signal_sense'])
                new_gene['node_types']['input'].append(new_type)
            total_nodes = new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']
            if total_nodes > new_gene['n_input']:
                for _ in range(random.randint(1, 3)):
                    to_node = random.randint(new_gene['n_input'], total_nodes - 1)
                    new_gene['connections'].append([new_gene['n_input'] - 1, to_node, random.uniform(-2, 2)])
            mutations_occurred.append('add_input_node')
        
        if random.random() < MUTATION_PROBABILITY['del_node'] * 0.5 and new_gene['n_input'] > 0:
            del_node_idx = random.randint(0, new_gene['n_input'] - 1)
            new_gene['connections'] = [c for c in new_gene['connections'] if c[0] != del_node_idx]
            for conn in new_gene['connections']:
                if conn[0] > del_node_idx: conn[0] -= 1
                if conn[1] > del_node_idx: conn[1] -= 1
            new_gene['n_input'] -= 1
            if 'node_types' in new_gene: new_gene['node_types']['input'].pop(del_node_idx)
            mutations_occurred.append('delete_input_node')
        
        if random.random() < MUTATION_PROBABILITY['add_node'] * 0.5:
            output_start = new_gene['n_input'] + new_gene['n_hidden']
            new_output_idx = output_start + new_gene['n_output']
            if output_start > 0:
                for _ in range(random.randint(1, 3)):
                    from_node = random.randint(0, output_start - 1)
                    new_gene['connections'].append([from_node, new_output_idx, random.uniform(-2, 2)])
            new_gene['n_output'] += 1
            if 'node_types' in new_gene:
                # 【【【 核心修正点 1 】】】
                # 将 'movement' 修正为 'move_vector'
                new_type = random.choice(['move_vector', 'signal'])
                new_gene['node_types']['output'].append(new_type)
            mutations_occurred.append('add_output_node')
        
        if random.random() < MUTATION_PROBABILITY['del_node'] * 0.5 and new_gene['n_output'] > 0:
            output_start = new_gene['n_input'] + new_gene['n_hidden']
            del_node_idx = output_start + random.randint(0, new_gene['n_output'] - 1)
            new_gene['connections'] = [c for c in new_gene['connections'] if c[1] != del_node_idx]
            for conn in new_gene['connections']:
                if conn[1] > del_node_idx: conn[1] -= 1
            new_gene['n_output'] -= 1
            if 'node_types' in new_gene:
                del_idx = del_node_idx - output_start
                if 0 <= del_idx < len(new_gene['node_types']['output']):
                    new_gene['node_types']['output'].pop(del_idx)
            mutations_occurred.append('delete_output_node')
        
        if random.random() < MUTATION_PROBABILITY['add_node']:
            hidden_start = new_gene['n_input']
            new_hidden_idx = hidden_start + new_gene['n_hidden']
            if hidden_start > 0:
                from_node = random.randint(0, hidden_start - 1)
                new_gene['connections'].append([from_node, new_hidden_idx, random.uniform(-2, 2)])
            total_nodes = new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']
            if new_hidden_idx + 1 < total_nodes:
                to_node = random.randint(new_hidden_idx + 1, total_nodes - 1)
                new_gene['connections'].append([new_hidden_idx, to_node, random.uniform(-2, 2)])
            new_gene['n_hidden'] += 1
            if 'node_types' in new_gene: new_gene['node_types']['hidden'].append('standard')
            mutations_occurred.append('add_hidden_node')
            
        if new_gene['n_hidden'] > 0 and random.random() < MUTATION_PROBABILITY['del_node']:
            hidden_start = new_gene['n_input']
            del_node_idx = hidden_start + random.randint(0, new_gene['n_hidden'] - 1)
            new_gene['connections'] = [c for c in new_gene['connections'] if c[0] != del_node_idx and c[1] != del_node_idx]
            for conn in new_gene['connections']:
                if conn[0] > del_node_idx: conn[0] -= 1
                if conn[1] > del_node_idx: conn[1] -= 1
            new_gene['n_hidden'] -= 1
            if 'node_types' in new_gene:
                del_idx = del_node_idx - hidden_start
                if 0 <= del_idx < len(new_gene['node_types']['hidden']):
                    new_gene['node_types']['hidden'].pop(del_idx)
            mutations_occurred.append('delete_hidden_node')

        if 'node_types' in new_gene and random.random() < MUTATION_PROBABILITY['point'] * 0.5:
            valid_categories = []
            for category in ['input', 'output', 'hidden']:
                if category in new_gene['node_types'] and len(new_gene['node_types'][category]) > 0:
                    valid_categories.append(category)
            if valid_categories:
                node_category = random.choice(valid_categories)
                node_idx = random.randint(0, len(new_gene['node_types'][node_category]) - 1)
                if node_category == 'input': new_type = random.choice(['field_sense', 'signal_sense'])
                # 【【【 核心修正点 2 】】】
                # 确保这里的列表也是正确的
                elif node_category == 'output': new_type = random.choice(['move_vector', 'signal'])
                else: new_type = 'standard'
                new_gene['node_types'][node_category][node_idx] = new_type
                mutations_occurred.append('node_type_mutation')

        is_mutant = len(mutations_occurred) > 0
        
        child = Agent(self.universe, self.logger, gene=new_gene, position=child_pos, 
                     energy=child_energy, parent_id=self.id, is_mutant=is_mutant)
        
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
            
            # ==================== 信号发射可视化修正区域 ====================
            # 绘制信号发射
            for i, activation in enumerate(self.last_action_vector):
                if abs(activation) > SIGNAL_RENDER_THRESHOLD:
                    # 获取此输出节点的正确类型
                    node_type = 'movement'  # 默认
                    if 'node_types' in self.gene and 'output' in self.gene['node_types'] and i < len(self.gene['node_types']['output']):
                        node_type = self.gene['node_types']['output'][i]
                    
                    # 只为“信号”类型的节点绘制可视化
                    if node_type == 'signal':
                        signal_radius = int(SIGNAL_EMISSION_RADIUS * abs(activation) * camera.zoom)
                        if signal_radius > 0:
                            # 根据信号影响的场来决定颜色，与update逻辑一致
                            signal_count = len(self.universe.fields) - 2  # 结果是2
                            field_idx = (i % signal_count) + 2  # 结果是 2 或 3
                            
                            # 默认颜色为蓝色 (对应Biotic 1)
                            signal_color = (0, 100, 255, 50) 
                            if field_idx == 3:  # 对应 Biotic 2
                                # 使用品红色以区别于红色的危险场
                                signal_color = (255, 0, 255, 50) 
                            
                            # 创建透明表面来绘制信号
                            signal_surface = pygame.Surface((signal_radius * 2, signal_radius * 2), pygame.SRCALPHA)
                            pygame.draw.circle(signal_surface, signal_color, (signal_radius, signal_radius), signal_radius)
                            # 绘制到主表面
                            surface.blit(signal_surface, (screen_pos[0] - signal_radius, screen_pos[1] - signal_radius))
            # ==================== 修正区域结束 ====================

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
    

    def _resolve_physics_and_collisions(self):
        """
        [管理者函数] 统一的物理引擎。
        这个函数负责准备数据，并调用Numba编译的核心计算函数。
        """
        living_agents = [agent for agent in self.agents if not agent.is_dead]
        if len(living_agents) < 2:
            return

        # 1. 准备Numba能理解的数据：从对象列表中提取纯粹的NumPy数组
        positions = np.array([agent.position for agent in living_agents], dtype=np.float32)
        radii = np.array([agent.radius for agent in living_agents], dtype=np.float32)
        
        # 2. 调用被@njit编译后的核心计算函数
        #    注意：这里传入的都是NumPy数组和基本数值，没有传入'self'
        new_positions = self._numba_collision_kernel(
            positions, 
            radii, 
            COLLISION_ITERATIONS,  # 使用全局参数
            WORLD_SIZE             # 使用全局参数
        )

        # 3. 将计算结果写回智能体对象
        for i, agent in enumerate(living_agents):
            agent.position.x = new_positions[i, 0]
            agent.position.y = new_positions[i, 1]

    @staticmethod
    @njit
    def _numba_collision_kernel(positions, radii, iterations, world_size):
        """
        [计算核心] - 这个函数会被Numba编译成高效机器码。
        它不依赖任何类实例(self)，只处理传入的NumPy数组和数值。
        """
        num_agents = len(positions)
        # 迭代数次以确保所有连锁碰撞都得到解决
        for _ in range(iterations):
            collision_found = False
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # Numba会将这些数学运算编译成高效的机器码
                    dist_vec_x = positions[i, 0] - positions[j, 0]
                    dist_vec_y = positions[i, 1] - positions[j, 1]

                    # 处理周期性边界
                    if abs(dist_vec_x) > world_size / 2:
                        dist_vec_x -= math.copysign(world_size, dist_vec_x)
                    if abs(dist_vec_y) > world_size / 2:
                        dist_vec_y -= math.copysign(world_size, dist_vec_y)
                        
                    dist_sq = dist_vec_x**2 + dist_vec_y**2
                    min_dist = radii[i] + radii[j]
                    
                    # 如果发生重叠
                    if dist_sq < min_dist**2 and dist_sq > 1e-9: # 增加一个小的阈值避免完全重合
                        collision_found = True
                        distance = math.sqrt(dist_sq)
                        overlap = min_dist - distance
                        
                        push_magnitude = overlap / 2.0
                        
                        # 标准化推开向量并应用推力
                        push_vec_x = (dist_vec_x / distance) * push_magnitude
                        push_vec_y = (dist_vec_y / distance) * push_magnitude
                        
                        positions[i, 0] += push_vec_x
                        positions[i, 1] += push_vec_y
                        positions[j, 0] -= push_vec_x
                        positions[j, 1] -= push_vec_y
                        
                        # 应用世界边界环绕
                        positions[i, 0] %= world_size
                        positions[i, 1] %= world_size
                        positions[j, 0] %= world_size
                        positions[j, 1] %= world_size
            
            if not collision_found:
                break
        return positions

    def update(self, dt):
        """更新宇宙状态"""
        if self.perf_monitor:
            self.perf_monitor.start_update()
            
        self.frame_count += 1
        
        # 1. 更新所有场
        self._update_fields_parallel(dt)

        # 2. 更新空间网格，用于邻居查找
        self.update_spatial_grid()
        
        # 3. 更新所有智能体的AI、能量、移动意图等
        #    (此时允许它们暂时重叠)
        updated_agents = []
        agent_batches = [self.agents[i:i+BATCH_SIZE] for i in range(0, len(self.agents), BATCH_SIZE)]
        future_results = [self.thread_pool.submit(self._update_agent_batch, batch, dt) for batch in agent_batches]
        for future in future_results:
            updated_agents.extend(future.result())
        self.agents = updated_agents
        
        # 4. 【调用统一物理引擎】强制解决所有物理问题
        self._resolve_physics_and_collisions()
        
        # 5. 处理繁殖 (在物理位置最终确定后)
        agent_batches_for_repro = [self.agents[i:i+BATCH_SIZE] for i in range(0, len(self.agents), BATCH_SIZE)]
        future_results = [self.thread_pool.submit(self._process_reproduction, batch) for batch in agent_batches_for_repro]
        new_children = []
        for future in future_results:
            new_children.extend(future.result())
        self.agents.extend(new_children)

        # 6. 处理种群数量控制 (出生和淘汰)
        if len(self.agents) < MIN_AGENTS_TO_SPAWN:
            self._spawn_new_agents()
        if len(self.agents) > MAX_AGENTS:
            self._cull_excess_agents() # 您可以在这里选择您喜欢的淘汰策略
            
        # 7. 定期记录状态
        if self.frame_count % 20 == 0:
            self.logger.log_state(self.frame_count, self.agents)
            self.logger.log_field(self.frame_count, self.fields)
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
    
    if agent is None:
        text = font.render("点击一个生命体来观察", True, (200, 200, 200))
        surface.blit(text, (panel_x + 20, 20))
        return
    
    y_offset = 20
    def draw_text(text, value, color=(255, 255, 255)):
        nonlocal y_offset
        text_surf = font.render(f"{text}: {value}", True, color)
        surface.blit(text_surf, (panel_x + 20, y_offset))
        y_offset += 25

    if agent.is_mutant:
        draw_text("观察对象 ID", f"{agent.id} (M)", (255, 255, 100))
    else:
        draw_text("观察对象 ID", agent.id, (100, 255, 100))

    draw_text("亲代 ID", agent.parent_id if agent.parent_id else "N/A")
    draw_text("基因型 ID", agent.genotype_id)
    # draw_text("年龄", agent.age)
    draw_text("能量 (E)", f"{agent.energy:.2f}")
    draw_text("位置 (p)", f"({agent.position.x:.1f}, {agent.position.y:.1f})")
    
    y_offset += 10
    draw_text("--- 基因特性 ---", "")
    draw_text("复杂度 (Ω)", f"{agent.complexity:.2f}")
    draw_text("输入/隐藏/输出", f"{agent.n_input}/{agent.n_hidden}/{agent.n_output}")
    draw_text("连接数", len(agent.gene['connections']))
    draw_text("环境吸收系数", f"{agent.env_absorption_coeff:.2f}")
    
    y_offset += 10
    draw_text("--- 生态特性 ---", "")
    id_value = round(agent.identity_vector, 2)
    r = 100 + abs(id_value) * 100
    g = 100 + (1 - abs(id_value)) * 100
    b = 200 - abs(id_value) * 100
    safe_color = (max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b))))
    draw_text("身份向量", f"{id_value:.2f}", safe_color)
    
    y_offset += 10
    draw_text("--- 行为输出 ---", "")
    
    output_types = agent.gene.get('node_types', {}).get('output', [])
    move_vec_count = 0
    signal_count = 0
    
    move_components = []
    for i, activation in enumerate(agent.last_action_vector):
        node_type = output_types[i] if i < len(output_types) else 'signal'
        if node_type == 'move_vector':
            move_components.append(activation)

    # 【【【 核心修正点：正确显示奇数和偶数个移动节点 】】】
    for i in range(0, len(move_components), 2):
        move_vec_count += 1
        vx = move_components[i]
        # 如果存在能配对的Y分量，则显示完整向量
        if i + 1 < len(move_components):
            vy = move_components[i+1]
            draw_text(f"移动向量 {move_vec_count}", f"({vx:.2f}, {vy:.2f})")
        else:
            # 如果落单了，则只显示X分量
            draw_text(f"移动向量 {move_vec_count}_X", f"{vx:.2f}")

    for i, activation in enumerate(agent.last_action_vector):
        node_type = output_types[i] if i < len(output_types) else 'signal'
        if node_type == 'signal':
            signal_count += 1
            draw_text(f"信号 {signal_count} 强度", f"{abs(activation):.2f}")
    
    y_offset += 20
    draw_neural_network(surface, font, agent, panel_x + 20, y_offset, panel_width - 40, 350, mouse_pos)

def draw_neural_network(surface, font, agent, x, y, width, height, mouse_pos):
    """绘制神经网络可视化"""
    title = font.render("计算核心 (Cᵢ) 拓扑图:", True, (200, 200, 100))
    surface.blit(title, (x, y))
    y += 30
    
    n_in, n_hid, n_out = agent.n_input, agent.n_hidden, agent.n_output
    
    # --- 修正后的标签和布局逻辑 ---
    
    # 1. 修正输入节点标签逻辑
    perception_channel_names = ["Energy_v", "Energy_gx", "Energy_gy", "Hazard_v", "Hazard_gx", "Hazard_gy"]
    signal_names = [f.name for f in agent.universe.fields if "Biotic" in f.name]
    for sig_name in signal_names:
        perception_channel_names.extend([f"{sig_name}_v", f"{sig_name}_gx", f"{sig_name}_gy"])
    
    input_labels = []
    for i in range(n_in):
        input_labels.append(perception_channel_names[i] if i < len(perception_channel_names) else f"Input_{i}")
        
    # 2. 动态生成输出节点标签
    output_labels = []
    output_types = agent.gene.get('node_types', {}).get('output', [])
    move_vec_count = 0
    signal_count = 0
    temp_move_tracker = 0 # 用于追踪移动向量的X/Y分量

    for i in range(n_out):
        node_type = output_types[i] if i < len(output_types) else 'signal'
        if node_type == 'move_vector':
            if temp_move_tracker % 2 == 0:
                move_vec_count += 1
                output_labels.append(f"MVec{move_vec_count}_x")
            else:
                output_labels.append(f"MVec{move_vec_count}_y")
            temp_move_tracker += 1
        elif node_type == 'signal':
            signal_count += 1
            output_labels.append(f"Sig_{signal_count}")

    # 3. 调整节点列的布局，为标签留出更多空间
    col_x = [x + 70, x + width // 2, x + width - 70]
    
    # --- 后续绘制逻辑 ---
    layers = [n_in, n_hid, n_out]
    node_positions = {}
    col_map = [0, 1, 2] if n_hid > 0 else [0, 2] 
    current_node_idx = 0
    visible_layer_idx = 0 
    for i, n_nodes in enumerate(layers):
        if n_nodes == 0: continue
        layer_y_start = y + (height - (n_nodes - 1) * 25) / 2 if n_nodes > 1 else y + height / 2
        for j in range(n_nodes):
            node_id = current_node_idx + j
            column_to_use = col_x[col_map[visible_layer_idx]]
            node_positions[node_id] = (int(column_to_use), int(layer_y_start + j * 25))
        current_node_idx += n_nodes
        visible_layer_idx += 1

    for from_n, to_n, weight in agent.gene['connections']:
        if from_n in node_positions and to_n in node_positions:
            start_pos, end_pos = node_positions[from_n], node_positions[to_n]
            line_width = min(3, max(1, abs(int(weight * 2))))
            color = (0, min(255, 100 + int(abs(weight) * 80)), 0) if weight > 0 else (min(255, 150 + int(abs(weight) * 50)), 50, 50)
            pygame.draw.line(surface, color, start_pos, end_pos, line_width)
    
    hover_info = None
    for node_id, pos in node_positions.items():
        is_input = node_id < n_in
        is_hidden = n_in <= node_id < n_in + n_hid
        
        # 颜色和节点绘制逻辑不变
        color = (255, 165, 0)
        if is_input: color = (100, 100, 255)
        elif not is_hidden:
            output_idx = node_id - (n_in + n_hid)
            node_type = output_types[output_idx] if output_idx < len(output_types) else 'signal'
            color = (255, 255, 100) if node_type == 'move_vector' else (100, 255, 100)
        
        activation = agent.node_activations[node_id]
        brightness = max(0, min(255, 128 + int(activation * 127)))
        color = tuple(min(255, c * brightness // 128) for c in color)
        radius = 6
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, (0,0,0), pos, radius, 1)

        # 标签获取和绘制
        label = None
        if is_input and node_id < len(input_labels):
            label = input_labels[node_id]
        elif not is_hidden:
            output_idx = node_id - (n_in + n_hid)
            if output_idx < len(output_labels):
                label = output_labels[output_idx]

        if label:
            label_surf = font.render(label, True, (200, 200, 200))
            # 4. 调整输入节点标签的绘制位置，避免遮挡
            if is_input:
                surface.blit(label_surf, (pos[0] - label_surf.get_width() - 10, pos[1] - 8))
            else: # 输出节点
                surface.blit(label_surf, (pos[0] + 10, pos[1] - 8))
        
        if math.hypot(mouse_pos[0] - pos[0], mouse_pos[1] - pos[1]) < radius:
            node_type_str = "隐藏"
            if is_input: node_type_str = "输入"
            elif not is_hidden: node_type_str = "输出"
            hover_info = (f"{node_type_str}节点 {node_id}", f"激活值: {agent.node_activations[node_id]:.3f}", mouse_pos)

    if hover_info:
        title, value, pos = hover_info
        title_surf = font.render(title, True, (255, 255, 255))
        value_surf = font.render(value, True, (255, 255, 255))
        box_rect = pygame.Rect(pos[0] + 10, pos[1] + 10, max(title_surf.get_width(), value_surf.get_width()) + 20, 50)
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
    parser = argparse.ArgumentParser(description="涌现认知生态系统 (ECE) v5.0")
    parser.add_argument("--no-gui", action="store_true", help="无GUI模式，仅运行计算")
    parser.add_argument("--continue-from", type=str, help="从指定的日志目录继续模拟")
    args = parser.parse_args()
    
    use_gui = not args.no_gui
    continue_simulation = bool(args.continue_from)

    # <<< 1. 创建日志目录、队列和独立的日志进程 >>>
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_queue = Queue()
    
    # 定义日志文件的表头
    headers = {
        'state': ["frame", "agent_id", "parent_id", "genotype_id", "is_mutant", "energy", 
                  "pos_x", "pos_y", "n_hidden", "n_connections", "gene_string"],
        'event': ["frame", "event_type", "details"],
        'field': ["frame", "field_type", "data"]
    }

    log_proc = multiprocessing.Process(
        target=logging_process_worker,
        args=(log_queue, log_dir, args.continue_from, headers),
        daemon=True # 设置为守护进程，主进程退出时它也会退出
    )
    log_proc.start()

    # <<< 2. 将队列传递给DataLogger >>>
    logger = DataLogger(log_queue, args.continue_from)
    
    if use_gui:
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        pygame.display.set_caption("涌现认知生态系统 (ECE) v5.0 - 高性能版")
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT), flags)
        clock = pygame.time.Clock()
        try: 
            font = pygame.font.SysFont("simhei", 16)
        except pygame.error: 
            font = pygame.font.SysFont(None, 22)
        current_screen_width, current_screen_height = screen.get_size()
        sim_area_width = current_screen_width - INFO_PANEL_WIDTH
    else:
        print("以无GUI模式运行，仅进行计算...")
        sim_area_width = INITIAL_SCREEN_WIDTH - INFO_PANEL_WIDTH
        current_screen_height = INITIAL_SCREEN_HEIGHT
    
    universe = Universe(logger, sim_area_width, current_screen_height, use_gui, continue_simulation)
    
    if not continue_simulation:
        logger.log_event(0, 'SIM_START', {'initial_agents': INITIAL_AGENT_COUNT, 'world_size': WORLD_SIZE, 'gui_mode': use_gui})
    else:
        logger.log_event(universe.frame_count, 'SIM_CONTINUE', {'agents': len(universe.agents), 'from_frame': universe.frame_count})
    
    running = True
    paused = False
    last_performance_update = 0
    render_every_n_frames = DEFAULT_RENDER_SKIP
    frame_counter = 0
    
    try:
        while running:
            frame_counter += 1
            render_this_frame = frame_counter % render_every_n_frames == 0
            
            if universe.perf_monitor and render_this_frame:
                universe.perf_monitor.start_frame()
                
            if use_gui:
                mouse_pos = pygame.mouse.get_pos()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False
                    if event.type == pygame.VIDEORESIZE:
                        current_screen_width, current_screen_height = event.size
                    universe.camera.handle_event(event, mouse_pos)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1 and mouse_pos[0] < universe.camera.render_width:
                            universe.handle_click(event.pos)
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE: paused = not paused
                        if event.key == pygame.K_RIGHT and paused: universe.update(0.016)
                        if event.key == pygame.K_F11: pygame.display.toggle_fullscreen()
                        if event.key == pygame.K_0: universe.view_mode = 0
                        elif event.key == pygame.K_1: universe.view_mode = 1
                        elif event.key == pygame.K_2: universe.view_mode = 2
                        elif event.key == pygame.K_3: universe.view_mode = 3
                        elif event.key == pygame.K_4: universe.view_mode = 4
                        elif event.key == pygame.K_F1: render_every_n_frames = 1
                        elif event.key == pygame.K_F2: render_every_n_frames = 2
                        elif event.key == pygame.K_F3: render_every_n_frames = 3
            else:
                if universe.frame_count % 100 == 0:
                    total_biomass = sum(agent.energy for agent in universe.agents)
                    print(f"帧: {universe.frame_count} | 生命体: {len(universe.agents)}/{MAX_AGENTS} | 总生物量: {int(total_biomass)}")
            
            if not paused:
                universe.update(0.016)
            
            if use_gui and render_this_frame:
                screen.fill((0,0,0))
                current_screen_width, current_screen_height = screen.get_size()
                sim_area_width = current_screen_width - INFO_PANEL_WIDTH
                if sim_area_width < 400: sim_area_width = 400
                info_panel_width = current_screen_width - sim_area_width
                universe.camera.update_render_size(sim_area_width, current_screen_height)
                sim_surface = pygame.Surface((sim_area_width, current_screen_height))
                universe.draw(screen, sim_surface)
                draw_inspector_panel(screen, font, universe.selected_agent, mouse_pos, sim_area_width, info_panel_width, current_screen_height)
                view_name = "全部"
                if 1 <= universe.view_mode <= len(universe.fields):
                    view_name = universe.fields[universe.view_mode - 1].name
                total_biomass = sum(agent.energy for agent in universe.agents)
                performance_info = ""
                if universe.perf_monitor and universe.frame_count - last_performance_update > UPDATE_INTERVAL:
                    stats = universe.perf_monitor.get_stats()
                    performance_info = f" | FPS: {stats['fps']} | 更新: {stats['update_ms']}ms | 渲染: {stats['render_ms']}ms"
                    last_performance_update = universe.frame_count
                info_text = f"帧: {universe.frame_count} | 生命体: {len(universe.agents)}/{MAX_AGENTS} ({universe.next_genotype_id}个基因型) | 总生物量: {int(total_biomass)} | 视图(0-4): {view_name}{performance_info} | {'[已暂停]' if paused else ''}"
                text_surface = font.render(info_text, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10))
                if render_every_n_frames > 1:
                    render_text = f"渲染频率: 每{render_every_n_frames}帧 (F1-F3调整)"
                    render_surface = font.render(render_text, True, (255, 200, 100))
                    screen.blit(render_surface, (10, 30))
                pygame.display.flip()
                if universe.perf_monitor:
                    universe.perf_monitor.end_frame(len(universe.agents))
    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
        running = False
    finally:
        # <<< 3. 通知日志进程结束并等待它完成 >>>
        print("主循环结束，正在等待日志进程完成...")
        log_queue.put(None)  # 发送结束信号
        log_proc.join(timeout=10) # 等待日志进程最多10秒
        if log_proc.is_alive():
            print("日志进程超时，强制终止。")
            log_proc.terminate()
        
        if use_gui:
            pygame.quit()
        print("模拟结束")

if __name__ == '__main__':
    main() 