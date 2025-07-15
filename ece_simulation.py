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
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pygame.math import Vector2

# --- 第一部分: 宇宙公理 (Axioms of the Universe) ---

# 1. 宇宙设定
INITIAL_SCREEN_WIDTH = 1200 
INITIAL_SCREEN_HEIGHT = 800
WORLD_SIZE = 512
INFO_PANEL_WIDTH = 400

# 2. 演化引擎参数
INITIAL_AGENT_COUNT = 500  # 增加初始智能体数量
MAX_AGENTS = 500          # 限制最大智能体数量为500
MIN_AGENTS_TO_SPAWN = 300
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

# 4. 性能优化参数
MAX_THREADS = max(4, multiprocessing.cpu_count() - 1)  # 使用CPU核心数-1的线程数
BATCH_SIZE = 100  # 每个批次处理的智能体数量
GRID_CELL_SIZE_FACTOR = 1.2  # 网格大小因子，用于空间划分优化
PERFORMANCE_MONITOR = True  # 启用性能监控
UPDATE_INTERVAL = 60  # 性能统计更新间隔（帧数）

# --- 数据日志系统 ---
class DataLogger:
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("logs", f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.state_log_path = os.path.join(self.log_dir, "simulation_log.csv")
        self.state_header = ["frame", "agent_id", "parent_id", "genotype_id", "is_mutant", "energy", 
                            "pos_x", "pos_y", "n_hidden", "n_connections", "computation_depth", "gene_string"]
        with open(self.state_log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(self.state_header)

        self.event_log_path = os.path.join(self.log_dir, "event_log.csv")
        self.event_header = ["frame", "event_type", "details"]
        with open(self.event_log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(self.event_header)
            
        self.agent_id_counter = 0

    def get_new_agent_id(self):
        self.agent_id_counter += 1
        return self.agent_id_counter

    def log_state(self, frame_number, agents):
        with open(self.state_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for agent in agents:
                gene_str = str(agent.gene)
                row = [frame_number, agent.id, agent.parent_id, agent.genotype_id, agent.is_mutant, 
                      round(agent.energy, 2), round(agent.position.x, 2), round(agent.position.y, 2), 
                      agent.gene['n_hidden'], len(agent.gene['connections']), agent.gene['computation_depth'], gene_str]
                writer.writerow(row)

    def log_event(self, frame, event_type, details):
        with open(self.event_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            details_str = json.dumps(details)
            writer.writerow([frame, event_type, details_str])

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

    def update(self, dt):
        # 移除扩散，只保留衰减 - 使用向量化操作提高性能
        self.grid *= (1 - FIELD_DECAY_RATE * dt)
        # 确保场值在0-1范围内
        np.clip(self.grid, 0, 1, out=self.grid)

    def get_value_and_gradient(self, pos):
        # 在给定位置获取场值和梯度
        x, y = int(pos.x % self.size), int(pos.y % self.size)
        val_up = self.grid[(y - 1) % self.size, x]
        val_down = self.grid[(y + 1) % self.size, x]
        val_left = self.grid[y, (x - 1) % self.size]
        val_right = self.grid[y, (x + 1) % self.size]
        value = self.grid[y, x]
        gradient = Vector2(val_right - val_left, val_down - val_up)
        return value, gradient

    def add_circular_source(self, pos, radius, value):
        # 在场中添加一个圆形源
        x_center, y_center = int(pos.x), int(pos.y)
        radius = int(radius)
        if radius <= 0: return

        # 使用网格化计算以提高性能
        y_range = np.arange(-radius, radius + 1)
        x_range = np.arange(-radius, radius + 1)
        full_x_coords, full_y_coords = np.meshgrid(x_range, y_range)
        
        # 创建圆形掩码
        mask = full_x_coords**2 + full_y_coords**2 <= radius**2
        
        x_coords_masked = full_x_coords[mask]
        y_coords_masked = full_y_coords[mask]
        
        # 计算梯度值（基于到中心的距离）- 使用更缓和的梯度
        distances = np.sqrt(x_coords_masked**2 + y_coords_masked**2)
        gradient_values = value * np.maximum(0, 1 - (distances / radius) ** ENERGY_GRADIENT_FACTOR)

        # 应用周期性边界条件
        x_indices_abs = (x_center + x_coords_masked) % self.size
        y_indices_abs = (y_center + y_coords_masked) % self.size
        
        # 确保场值不超过1.0
        current_values_at_target = self.grid[y_indices_abs, x_indices_abs]
        room_to_add = np.maximum(0, 1.0 - current_values_at_target)
        values_to_add = np.minimum(gradient_values, room_to_add)

        self.grid[y_indices_abs, x_indices_abs] += values_to_add

    def draw(self, surface, camera, alpha=128):
        # 仅绘制可见区域以提高性能
        render_width, render_height = camera.render_width, camera.render_height
        top_left = camera.screen_to_world((0,0))
        bottom_right = camera.screen_to_world((render_width, render_height))
        start_x = max(0, int(top_left.x))
        start_y = max(0, int(top_left.y))
        end_x = min(self.size, int(bottom_right.x) + 2)
        end_y = min(self.size, int(bottom_right.y) + 2)

        if start_x >= end_x or start_y >= end_y: return

        # 提取可见区域的子网格
        sub_grid = self.grid[start_y:end_y, start_x:end_x]
        
        # 创建彩色数组（仅填充特定颜色通道）
        color_array = np.zeros((sub_grid.shape[1], sub_grid.shape[0], 3), dtype=np.uint8)
        color_array[:, :, self.color] = (sub_grid.T * 255).astype(np.uint8)
        
        # 创建表面并设置透明度
        render_surface = pygame.surfarray.make_surface(color_array)
        render_surface.set_alpha(alpha)
        
        # 计算屏幕位置和缩放尺寸
        screen_pos = camera.world_to_screen(Vector2(start_x, start_y))
        scaled_size = (int(sub_grid.shape[1] * camera.zoom), int(sub_grid.shape[0] * camera.zoom))
        
        # 渲染到屏幕
        if scaled_size[0] > 0 and scaled_size[1] > 0:
            surface.blit(pygame.transform.scale(render_surface, scaled_size), screen_pos) 

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
            input_types.append(random.choice(['general', 'field_sense', 'signal_sense', 'special']))
            
        output_types = []
        for _ in range(n_output):
            output_types.append(random.choice(['movement', 'signal', 'energy', 'special']))
            
        hidden_types = []
        for _ in range(n_hidden):
            hidden_types.append(random.choice(['standard', 'memory', 'processing', 'special']))
        
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
        # 从基因构建计算核心
        self.n_input = self.gene['n_input']
        self.n_output = self.gene['n_output']
        self.n_hidden = self.gene['n_hidden']
        self.total_nodes = self.n_input + self.n_hidden + self.n_output
        self.computation_depth = self.gene['computation_depth']
        
        # 节点激活向量 - 不需要特殊处理，自然会作为计算状态保留
        self.node_activations = np.zeros(self.total_nodes, dtype=np.float32)
        
        # 构建连接矩阵
        self.connection_matrix = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for from_n, to_n, weight in self.gene['connections']:
            if 0 <= from_n < self.total_nodes and self.n_input <= to_n < self.total_nodes:
                self.connection_matrix[to_n, from_n] = weight
                
        # 计算复杂度和其他基因表达
        self.complexity = len(self.gene['connections']) + self.n_hidden * 2
        self.e_repro = 150 + self.complexity * 2  # 繁殖阈值
        self.e_child = 50 + self.complexity  # 子代初始能量
        self.e_res = 20 + self.complexity * 0.5  # 死亡后回馈环境的能量
        self.metabolism_cost = 0.15 + 0.0005 * (self.complexity**2)  # 基础代谢成本
        
        # 基因表达的身份向量（用于相互作用和生态位定位）
        if self.gene['connections']:
            weights = [c[2] for c in self.gene['connections']]
            self.identity_vector = np.mean(weights)
        else:
            self.identity_vector = 0
        
        # 环境吸收系数
        self.env_absorption_coeff = self.gene.get('env_absorption_coeff', 0.5)
        
        # 注册基因型ID
        self.genotype_id = self.universe.get_or_create_genotype_id(self.gene)

    def update(self, dt, neighbors):
        if self.is_dead: 
            return
        
        # 1. 感知与决策 - 实现"计算的有限深度"法则
        # 获取环境感知向量
        perception_vector = self.universe.get_perception_vector(self.position)
        
        # 根据节点类型决定如何处理输入
        for i in range(min(self.n_input, len(perception_vector))):
            # 获取当前节点类型
            node_type = 'general'  # 默认类型
            if 'node_types' in self.gene and i < len(self.gene['node_types']['input']):
                node_type = self.gene['node_types']['input'][i]
            
            # 根据节点类型处理输入
            if node_type == 'field_sense' and i < len(perception_vector):
                # 场感知节点直接使用感知向量
                self.node_activations[i] = perception_vector[i]
            elif node_type == 'signal_sense':
                # 信号感知节点有一定概率保持其当前值（记忆效应）
                if random.random() > 0.7:  # 70%概率更新
                    if i < len(perception_vector):
                        self.node_activations[i] = perception_vector[i]
            elif node_type == 'special':
                # 特殊节点值随机波动，创造突发性行为
                self.node_activations[i] = self.node_activations[i] * 0.9 + random.uniform(-0.1, 0.1)
            else:  # 'general'或其他
                # 一般节点直接使用感知向量或保持原值
                if i < len(perception_vector):
                    self.node_activations[i] = perception_vector[i]
        
        # 执行计算步骤（由基因决定的深度）- 使用矩阵运算提高效率
        for _ in range(self.computation_depth):
            # 计算新的激活值
            inputs = np.dot(self.connection_matrix.T, self.node_activations)
            new_activations = np.tanh(inputs)
            
            # 更新隐藏层和输出层的激活值
            self.node_activations[self.n_input:] = new_activations[self.n_input:]
        
        # 读取当前输出层的值作为行动指令
        output_activations = self.node_activations[-self.n_output:]
        self.last_action_vector = output_activations
        
        # 初始化行为向量
        move_vector = Vector2(0, 0)
        
        # 根据输出节点类型决定行为
        for i, activation in enumerate(output_activations):
            # 获取当前输出节点类型
            node_type = 'movement'  # 默认为移动类型
            if 'node_types' in self.gene and i < len(self.gene['node_types']['output']):
                node_type = self.gene['node_types']['output'][i]
            
            # 根据节点类型执行不同行为
            if node_type == 'movement':
                # 移动节点影响移动向量
                # 每对节点控制一个方向
                if i % 2 == 0 and i+1 < len(output_activations):  # X方向
                    move_vector.x += activation
                elif i % 2 == 1:  # Y方向
                    move_vector.y += activation
            elif node_type == 'signal' and i < 4:  # 限制最多作用于前4个信号场
                # 信号节点控制信号释放
                field_idx = i % len(self.universe.fields)
                if abs(activation) > 0.1:
                    self.universe.fields[field_idx].add_circular_source(
                        self.position, SIGNAL_EMISSION_RADIUS, abs(activation) * 0.01)
            elif node_type == 'energy':
                # 能量调控节点 - 可以调整能量吸收效率
                self.env_absorption_coeff = activation * 0.1 + self.gene.get('env_absorption_coeff', 0.5)
            # 'special'节点不执行明确动作，但其值会传递给其他节点
        
        # 确保所有生物都有最小移动量
        if move_vector.length_squared() < MIN_MOVEMENT_JITTER**2:
            move_vector.x += random.uniform(-MIN_MOVEMENT_JITTER, MIN_MOVEMENT_JITTER)
            move_vector.y += random.uniform(-MIN_MOVEMENT_JITTER, MIN_MOVEMENT_JITTER)
        
        # 2. 移动
        self.position += move_vector * dt * MOVEMENT_SPEED_FACTOR

        # 3. 添加温和排斥力
        repulsion_vector = Vector2(0, 0)
        close_neighbors_count = 0
        overlapping_neighbors = 0
        
        # 收集所有邻居信息，以便更好地处理重叠
        neighbor_data = []
        for other in neighbors:
            if other is self: 
                continue
            dist_vec = self.position - other.position
            dist_sq = dist_vec.length_squared()
            min_dist = self.radius + other.radius
            
            # 收集邻居数据
            neighbor_data.append({
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
            for data in neighbor_data:
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
        
        # 4. 严格碰撞解决（多次迭代）- 增加对高密度区域的特殊处理
        for iteration in range(COLLISION_ITERATIONS):
            collision_occurred = False
            # 按距离排序，先处理最严重的重叠
            sorted_neighbors = sorted(neighbor_data, key=lambda x: x['dist_sq'])
            
            for data in sorted_neighbors:
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
                        
                    self.position += data['dist_vec'].normalize() * overlap * push_factor * (1.0 / COLLISION_ITERATIONS)
            
            # 如果没有碰撞发生，提前退出循环
            if not collision_occurred:
                break
                
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

    def reproduce(self):
        # 繁殖检查：能量必须达到繁殖阈值
        if self.energy < self.e_repro:
            return None

        # 获取所有可能的周围智能体
        neighbors = self.universe.get_neighbors(self)
        
        # 尝试找到一个没有重叠的位置
        max_attempts = 20  # 增加尝试次数从10到20
        child_pos = None
        min_safe_distance = self.radius * 3.0  # 增加安全距离从2倍半径+1到3倍半径
        
        # 缓存所有智能体位置，不仅仅是直接邻居
        all_positions = []
        for agent in self.universe.agents:
            if agent is not self and not agent.is_dead:
                all_positions.append(agent.position)
        
        for _ in range(max_attempts):
            # 生成一个候选位置
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(self.radius * 3.0, self.radius * 8.0)  # 确保至少在3倍半径距离外
            candidate_pos = Vector2(
                self.position.x + math.cos(angle) * distance,
                self.position.y + math.sin(angle) * distance
            )
            
            # 对周期性边界条件进行修正
            candidate_pos.x %= WORLD_SIZE
            candidate_pos.y %= WORLD_SIZE
            
            # 检查这个位置是否会与任何其他智能体重叠
            is_valid = True
            for pos in all_positions:
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
        
        # 如果找不到合适的位置，则不繁殖
        if child_pos is None:
            return None

        # 消耗能量创建后代
        self.energy -= self.e_child
        
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
            from_n = random.randint(0, new_gene['n_input'] + new_gene['n_hidden'] - 1)
            to_n = random.randint(new_gene['n_input'], new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output'] - 1)
            if to_n > from_n:  # 避免回路
                new_gene['connections'].append([from_n, to_n, random.uniform(-1, 1)])
                mutations_occurred.append('add_connection')
            
        # 删除连接
        if len(new_gene['connections']) > 2 and random.random() < MUTATION_PROBABILITY['del_conn']:
            # 保留至少2个连接，确保基本功能
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
            # 完全随机添加输入节点
            if new_gene['n_input'] < 20:  # 限制最大数量
                new_gene['n_input'] += 1
                
                # 更新节点类型记录
                if 'node_types' in new_gene:
                    # 随机选择新节点类型
                    new_type = random.choice(['general', 'field_sense', 'signal_sense', 'special'])
                    new_gene['node_types']['input'].append(new_type)
                
                # 为新节点创建随机连接
                for _ in range(random.randint(1, 3)):
                    to_node = random.randint(new_gene['n_input'], 
                                             new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output'] - 1)
                    new_gene['connections'].append([new_gene['n_input'] - 1, to_node, random.uniform(-2, 2)])
                
                mutations_occurred.append('add_input_node')
        
        # 2. 删除输入节点突变
        if random.random() < MUTATION_PROBABILITY['del_node'] * 0.5 and new_gene['n_input'] > 2:
            # 确保保留至少2个输入节点
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
            # 完全随机添加输出节点
            if new_gene['n_output'] < 12:  # 限制最大数量
                # 更新节点索引计算
                output_start = new_gene['n_input'] + new_gene['n_hidden']
                new_output_idx = output_start + new_gene['n_output']
                
                # 为新输出节点创建随机连接
                for _ in range(random.randint(1, 3)):
                    from_node = random.randint(0, output_start - 1)
                    new_gene['connections'].append([from_node, new_output_idx, random.uniform(-2, 2)])
                
                # 更新节点数量
                new_gene['n_output'] += 1
                
                # 更新节点类型记录
                if 'node_types' in new_gene:
                    # 随机选择新节点类型
                    new_type = random.choice(['movement', 'signal', 'energy', 'special'])
                    new_gene['node_types']['output'].append(new_type)
                
                mutations_occurred.append('add_output_node')
        
        # 4. 删除输出节点突变
        if random.random() < MUTATION_PROBABILITY['del_node'] * 0.5 and new_gene['n_output'] > 1:
            # 确保保留至少1个输出节点
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
            from_node = random.randint(0, hidden_start - 1)
            new_gene['connections'].append([from_node, new_hidden_idx, random.uniform(-2, 2)])
            
            # 输出连接
            to_node = random.randint(new_hidden_idx + 1, 
                                     new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output'])
            if to_node < new_gene['n_input'] + new_gene['n_hidden'] + new_gene['n_output']:
                new_gene['connections'].append([new_hidden_idx, to_node, random.uniform(-2, 2)])
            
            # 更新节点数量
            new_gene['n_hidden'] += 1
            
            # 更新节点类型记录
            if 'node_types' in new_gene:
                # 随机选择新节点类型
                new_type = random.choice(['standard', 'memory', 'processing', 'special'])
                new_gene['node_types']['hidden'].append(new_type)
            
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
            node_category = random.choice(['input', 'output', 'hidden'])
            if new_gene['node_types'][node_category]:
                # 随机选择该类别中的一个节点
                node_idx = random.randint(0, len(new_gene['node_types'][node_category]) - 1)
                
                # 根据类别提供不同的可能类型
                if node_category == 'input':
                    new_type = random.choice(['general', 'field_sense', 'signal_sense', 'special'])
                elif node_category == 'output':
                    new_type = random.choice(['movement', 'signal', 'energy', 'special'])
                else:  # hidden
                    new_type = random.choice(['standard', 'memory', 'processing', 'special'])
                
                # 应用新类型
                new_gene['node_types'][node_category][node_idx] = new_type
                mutations_occurred.append('node_type_mutation')

        # 检查是否发生了突变
        is_mutant_child = len(mutations_occurred) > 0
        
        # 创建子代（使用找到的无重叠位置）
        child = Agent(self.universe, self.logger, gene=new_gene, position=child_pos, 
                     energy=self.e_child, parent_id=self.id, is_mutant=is_mutant_child)
        
        # 记录突变事件
        if is_mutant_child:
            self.logger.log_event(
                self.universe.frame_count, 
                'MUTATION', 
                {'parent_id': self.id, 'parent_genotype': self.genotype_id, 
                 'child_id': child.id, 'child_genotype': child.genotype_id, 
                 'types': list(set(mutations_occurred))}
            )
        return child

    def draw(self, surface, camera):
        if self.is_dead: 
            return
            
        # 计算屏幕位置
        screen_pos = camera.world_to_screen(self.position)
        
        # 计算半径（考虑缩放）
        radius = max(1.0, camera.zoom * self.radius)
        
        # 如果不在屏幕上则跳过
        if not (-radius < screen_pos[0] < camera.render_width + radius and 
                -radius < screen_pos[1] < camera.render_height + radius):
            return
        
        # 基于基因型ID设置颜色
        hue = (self.genotype_id * 20) % 360
        color = pygame.Color(0)
        color.hsva = (hue, 85, 90, 100)
        
        # 绘制智能体
        pygame.draw.circle(surface, color, screen_pos, int(radius))
        
        # 为选中的智能体绘制特殊边框
        if self.universe.selected_agent and self.universe.selected_agent.id == self.id:
            pygame.draw.circle(surface, (255, 255, 0), screen_pos, int(radius + 2), 2)
            
            # 绘制信号发射可视化
            base_output = self.gene.get('base_output', 2)
            if len(self.last_action_vector) > base_output:  # 确保有信号输出
                for i, signal in enumerate(self.last_action_vector[base_output:]):
                    if abs(signal) > 0.1:  # 只绘制有效信号
                        # 信号强度决定圆圈大小
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
            pygame.draw.circle(surface, (255, 255, 255), screen_pos, int(radius), 1) 

# --- 宇宙系统 ---
class Universe:
    def __init__(self, logger, render_width, render_height):
        self.logger = logger
        
        # 初始化信息场
        self.fields = [
            Field(WORLD_SIZE, 1, "Nutrient/Energy"),  # 营养/能量场（绿色）
            Field(WORLD_SIZE, 0, "Hazard"),          # 危险/障碍场（红色）
            Field(WORLD_SIZE, 2, "Biotic 1"),        # 生物信号场1（蓝色）
            Field(WORLD_SIZE, 0, "Biotic 2"),        # 生物信号场2（红色）
        ]
        self.nutrient_field, self.hazard_field, self.biotic_field_1, self.biotic_field_2 = self.fields
        
        # 初始化宇宙状态
        self.frame_count = 0
        self.selected_agent = None
        self.view_mode = 1  # 默认显示营养场
        
        # 初始化相机
        self.camera = Camera(render_width, render_height)
        
        # 初始化空间网格（用于邻居查找优化）
        self.grid_cell_size = INTERACTION_RANGE * GRID_CELL_SIZE_FACTOR
        self.spatial_grid = defaultdict(list)
        
        # 基因型注册表
        self.genotype_registry = {}
        self.next_genotype_id = 0
        
        # 封闭能量系统：在模拟开始时一次性投放能量
        self._initial_energy_seeding()
        
        # 创建初始智能体 - 确保位置不重叠
        self.agents = []
        min_safe_distance = AGENT_RADIUS * 3.0  # 安全距离
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
                    
                    if dist_sq < min_safe_distance * min_safe_distance:
                        valid_position = False
                        break
                
                # 如果找到有效位置，创建智能体
                if valid_position:
                    # 创建新智能体并添加到列表
                    agent = Agent(self, self.logger, position=candidate_pos)
                    self.agents.append(agent)
                    occupied_positions.append(candidate_pos)
                    break
            
            # 如果无法找到有效位置，记录警告并尝试使用更小的安全距离
            if not valid_position and min_safe_distance > AGENT_RADIUS:
                min_safe_distance *= 0.9  # 逐步减小安全距离
                self.logger.log_event(0, 'SPAWN_WARNING', 
                                    {'message': f'Reducing safe distance to {min_safe_distance:.2f}'})
                
                # 再次尝试创建智能体
                for _ in range(max_attempts):
                    candidate_pos = Vector2(
                        random.uniform(0, WORLD_SIZE),
                        random.uniform(0, WORLD_SIZE)
                    )
                    
                    valid_position = True
                    for existing_pos in occupied_positions:
                        dx = min(abs(candidate_pos.x - existing_pos.x), WORLD_SIZE - abs(candidate_pos.x - existing_pos.x))
                        dy = min(abs(candidate_pos.y - existing_pos.y), WORLD_SIZE - abs(candidate_pos.y - existing_pos.y))
                        dist_sq = dx * dx + dy * dy
                        
                        if dist_sq < min_safe_distance * min_safe_distance:
                            valid_position = False
                            break
                    
                    if valid_position:
                        agent = Agent(self, self.logger, position=candidate_pos)
                        self.agents.append(agent)
                        occupied_positions.append(candidate_pos)
                        break
        
        # 记录实际创建的智能体数量
        actual_count = len(self.agents)
        if actual_count < INITIAL_AGENT_COUNT:
            self.logger.log_event(0, 'SPAWN_WARNING', 
                                {'message': f'Only created {actual_count}/{INITIAL_AGENT_COUNT} agents due to space constraints'})
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        
        # 性能监控
        self.perf_monitor = PerformanceMonitor() if PERFORMANCE_MONITOR else None

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
        """更新空间网格（用于邻居查找）"""
        self.spatial_grid.clear()
        for agent in self.agents:
            if not agent.is_dead:
                grid_x = int(agent.position.x / self.grid_cell_size)
                grid_y = int(agent.position.y / self.grid_cell_size)
                self.spatial_grid[(grid_x, grid_y)].append(agent)

    def get_neighbors(self, agent):
        """获取智能体的邻居（包括自身）"""
        neighbors = []
        grid_x = int(agent.position.x / self.grid_cell_size)
        grid_y = int(agent.position.y / self.grid_cell_size)
        grid_w = int(WORLD_SIZE / self.grid_cell_size)
        grid_h = int(WORLD_SIZE / self.grid_cell_size)
        
        # 检查九宫格邻居
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                wrapped_x = (grid_x + dx) % grid_w
                wrapped_y = (grid_y + dy) % grid_h
                neighbors.extend(self.spatial_grid.get((wrapped_x, wrapped_y), []))
                
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

        # 如果智能体数量过少，补充一些新的随机智能体
        if len(self.agents) < MIN_AGENTS_TO_SPAWN and self.frame_count % 10 == 0:
            new_agents = []
            for _ in range(10):
                # 尝试找到一个不重叠的位置
                max_attempts = 30  # 增加尝试次数
                new_pos = None
                min_safe_distance = AGENT_RADIUS * 3.0  # 增加安全距离
                
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
        
        # 如果智能体数量过多，淘汰一些能量最低的
        if len(self.agents) > MAX_AGENTS:
            self.agents.sort(key=lambda a: a.energy)
            num_to_remove = len(self.agents) - MAX_AGENTS
            culled_ids = [a.id for a in self.agents[:num_to_remove]]
            
            for agent_to_remove in self.agents[:num_to_remove]:
                agent_to_remove.is_dead = True
                self.on_agent_death(agent_to_remove)
                
            self.agents = self.agents[num_to_remove:]
            self.logger.log_event(self.frame_count, 'CULL', 
                                 {'count': num_to_remove, 'culled_ids': culled_ids})
            
        # 定期记录状态
        if self.frame_count % 20 == 0:
            self.logger.log_state(self.frame_count, self.agents)
            
        if self.perf_monitor:
            self.perf_monitor.end_update()

    def draw(self, surface, sim_surface):
        """绘制整个宇宙"""
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
            
        # 绘制所有智能体
        for agent in self.agents:
            agent.draw(sim_surface, self.camera)
            
        # 将模拟表面绘制到主表面
        surface.blit(sim_surface, (0, 0))
        
        if self.perf_monitor:
            self.perf_monitor.end_render()

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
    # 环境感知标签
    basic_input_labels = ["N_v", "N_gx", "N_gy", "H_v", "H_gx", "H_gy", "B1_v", "B1_gx", "B1_gy", "B2_v", "B2_gx", "B2_gy"]
    for i in range(n_in):
        if i < len(basic_input_labels):
            input_labels.append(basic_input_labels[i])
        else:
            # 额外输入节点
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
        output_labels.append(f"Sig_{i-1}")
    
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
        if self.frame_times:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def start_update(self):
        self.update_start_time = time.time()
    
    def end_update(self):
        update_time = time.time() - self.update_start_time
        self.update_times.append(update_time)
        if len(self.update_times) > 100:
            self.update_times.pop(0)
        self.avg_update_time = sum(self.update_times) / len(self.update_times)
    
    def start_render(self):
        self.render_start_time = time.time()
    
    def end_render(self):
        render_time = time.time() - self.render_start_time
        self.render_times.append(render_time)
        if len(self.render_times) > 100:
            self.render_times.pop(0)
        self.avg_render_time = sum(self.render_times) / len(self.render_times)
    
    def get_stats(self):
        return {
            'fps': round(self.fps, 1),
            'agents': self.agent_counts[-1] if self.agent_counts else 0,
            'update_ms': round(self.avg_update_time * 1000, 1),
            'render_ms': round(self.avg_render_time * 1000, 1)
        }

def main():
    """主函数"""
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
    
    clock = pygame.time.Clock()
    
    # 设置字体
    try: 
        font = pygame.font.SysFont("simhei", 16)
    except pygame.error: 
        font = pygame.font.SysFont(None, 22)
    
    # 初始化数据记录器
    logger = DataLogger()
    
    # 设置模拟区域大小
    current_screen_width, current_screen_height = screen.get_size()
    sim_area_width = current_screen_width - INFO_PANEL_WIDTH
    
    # 创建宇宙
    universe = Universe(logger, sim_area_width, current_screen_height)
    
    # 记录模拟开始事件
    logger.log_event(0, 'SIM_START', {'initial_agents': INITIAL_AGENT_COUNT, 'world_size': WORLD_SIZE})
    
    # 控制变量
    running = True
    paused = False
    last_performance_update = 0
    
    # 主循环
    while running:
        if universe.perf_monitor:
            universe.perf_monitor.start_frame()
            
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
        
        # 非暂停状态下更新模拟
        if not paused:
            # 使用固定的时间步长，提高模拟稳定性
            fixed_dt = 0.016  # 约60FPS
            universe.update(fixed_dt)
        
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
            
        info_text = f"帧: {universe.frame_count} | 生命体: {len(universe.agents)} ({universe.next_genotype_id}个基因型) | " \
                   f"总生物量: {int(total_biomass)} | 视图(0-4): {view_name}{performance_info} | {'[已暂停]' if paused else ''}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        # 更新显示 - 使用flip而不是update以利用硬件加速
        pygame.display.flip()
        
        # 使用clock.tick()而不是sleep，更精确地控制帧率
        # 不限制帧率，让CPU尽可能多地工作
        clock.tick(0)
        
        if universe.perf_monitor:
            universe.perf_monitor.end_frame(len(universe.agents))

    # 关闭线程池
    if hasattr(universe, 'thread_pool'):
        universe.thread_pool.shutdown()
        
    # 退出Pygame
    pygame.quit()

if __name__ == '__main__':
    main() 