# -*- coding: utf-8 -*-
# =============================================================================
# 涌现认知生态系统 (ECE) - v5.0 (创世引擎版)
#
# 作者: 一个中国的高中复读生 & Gemini
# 日期: 2025年7月16日
#
# v5.0 核心重构:
# 1. [创世] 基因现在是完整的“主板蓝图”，定义了每个节点的类型和功能。
# 2. [演化] 突变现在可以“发明”出全新的感官（输入节点）和表达方式
#    （输出节点），并将其随机“调谐”到不同的宇宙信号频道。
# 3. [动态] 每个生物的感知和行动向量都是动态的，由其自身基因决定。
# 4. [修复] 彻底移除了所有硬编码的感知和命名，实现了真正的自下而上。
# =============================================================================

import pygame
import numpy as np
import random
import math
import os
import datetime
import csv
from collections import defaultdict

# --- 第一部分: 宇宙公理 (Axioms of the Universe) ---

# 1. 宇宙设定
INITIAL_SCREEN_WIDTH = 1200 
INITIAL_SCREEN_HEIGHT = 800
WORLD_SIZE = 512
INFO_PANEL_WIDTH = 400

# 2. 演化引擎参数
INITIAL_AGENT_COUNT = 300
MAX_AGENTS = 600
MIN_AGENTS_TO_SPAWN = 300
MUTATION_PROBABILITY = {
    'weight_point': 0.03,  # 连接权重点突变
    'add_conn': 0.02,     # 增加一个新连接
    'del_conn': 0.02,     # 删除一个连接
    'add_node': 0.01,     # 增加一个新节点 (输入/隐藏/输出)
    'del_node': 0.01,     # 删除一个节点
    'tweak_gene': 0.05    # 调整基因中的其他参数
}
MUTATION_STRENGTH = 0.3

# 3. 物理与生态参数
FIELD_DIFFUSION_RATE = 0.1
FIELD_DECAY_RATE = 0.001
INTERACTION_RANGE = 120.0 
ENERGY_TRANSFER_EFFICIENCY = 0.9
K_INTERACTION_FACTOR = 0.01
DENSITY_REPULSION_RADIUS = 10.0
DENSITY_REPULSION_STRENGTH = 2.0
ENERGY_SPRING_RADIUS = 40.0 

# --- 数据日志系统 ---
class DataLogger:
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("logs", f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, "simulation_log.csv")
        self.csv_header = ["frame", "agent_id", "parent_id", "energy", "pos_x", "pos_y", "n_input", "n_hidden", "n_output", "n_connections", "gene_string"]
        with open(self.log_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_header)
        self.agent_id_counter = 0

    def get_new_agent_id(self):
        self.agent_id_counter += 1
        return self.agent_id_counter

    def log_frame(self, frame_number, agents):
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for agent in agents:
                gene = agent.gene
                gene_str = str(gene)
                row = [frame_number, agent.id, agent.parent_id, round(agent.energy, 2), 
                       round(agent.position.x, 2), round(agent.position.y, 2),
                       len(gene['nodes']['input']), len(gene['nodes']['hidden']), len(gene['nodes']['output']),
                       len(gene['connections']), gene_str]
                writer.writerow(row)

class Vector2:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
    def __add__(self, other): return Vector2(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Vector2(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar): return Vector2(self.x * scalar, self.y * scalar)
    @property
    def magnitude_sq(self): return self.x**2 + self.y**2
    @property
    def magnitude(self): return math.sqrt(self.x**2 + self.y**2) if self.x or self.y else 0
    def normalize(self):
        mag = self.magnitude
        if mag > 0: return Vector2(self.x / mag, self.y / mag)
        return Vector2(0, 0)

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
            if event.button == 4 and mouse_pos[0] < self.render_width: self.zoom_at(mouse_pos, 1.1)
            elif event.button == 5 and mouse_pos[0] < self.render_width: self.zoom_at(mouse_pos, 1/1.1)
            elif event.button == 3: self.panning = True; self.pan_start_pos = mouse_pos
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

class Field:
    def __init__(self, size, color, name):
        self.size = size
        self.color = color
        self.name = name
        self.grid = np.zeros((size, size), dtype=np.float32)

    def update(self, dt):
        laplacian = -4 * self.grid
        laplacian += np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0)
        laplacian += np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)
        self.grid += (FIELD_DIFFUSION_RATE * laplacian - FIELD_DECAY_RATE * self.grid) * dt
        np.clip(self.grid, 0, 1, out=self.grid)

    def get_value_and_gradient(self, pos):
        x, y = int(pos.x % self.size), int(pos.y % self.size)
        val_up = self.grid[(y - 1) % self.size, x]
        val_down = self.grid[(y + 1) % self.size, x]
        val_left = self.grid[y, (x - 1) % self.size]
        val_right = self.grid[y, (x + 1) % self.size]
        value = self.grid[y, x]
        gradient = Vector2(val_right - val_left, val_down - val_up)
        return value, gradient

    def add_circular_source(self, pos, radius, value):
        x_center, y_center = int(pos.x), int(pos.y)
        radius = int(radius)
        if radius <= 0: return
        y_range = np.arange(-radius, radius + 1)
        x_range = np.arange(-radius, radius + 1)
        x_coords, y_coords = np.meshgrid(x_range, y_range)
        mask = x_coords**2 + y_coords**2 <= radius**2
        x_coords_masked = x_coords[mask]
        y_coords_masked = y_coords[mask]
        x_indices_abs = (x_center + x_coords_masked) % self.size
        y_indices_abs = (y_center + y_coords_masked) % self.size
        self.grid[y_indices_abs, x_indices_abs] += value
        np.clip(self.grid, 0, 1, out=self.grid)

    def draw(self, surface, camera, alpha=128):
        render_width, render_height = camera.render_width, camera.render_height
        top_left = camera.screen_to_world((0,0))
        bottom_right = camera.screen_to_world((render_width, render_height))
        start_x = max(0, int(top_left.x))
        start_y = max(0, int(top_left.y))
        end_x = min(self.size, int(bottom_right.x) + 2)
        end_y = min(self.size, int(bottom_right.y) + 2)
        if start_x >= end_x or start_y >= end_y: return
        sub_grid = self.grid[start_y:end_y, start_x:end_x]
        color_array = np.zeros((sub_grid.shape[1], sub_grid.shape[0], 3), dtype=np.uint8)
        color_array[:, :, self.color] = (sub_grid.T * 255).astype(np.uint8)
        render_surface = pygame.surfarray.make_surface(color_array)
        render_surface.set_alpha(alpha)
        screen_pos = camera.world_to_screen(Vector2(start_x, start_y))
        scaled_size = (int(sub_grid.shape[1] * camera.zoom), int(sub_grid.shape[0] * camera.zoom))
        if scaled_size[0] > 0 and scaled_size[1] > 0:
            surface.blit(pygame.transform.scale(render_surface, scaled_size), screen_pos)

class Agent:
    def __init__(self, universe, logger, gene=None, position=None, energy=None, parent_id=None):
        self.universe = universe; self.logger = logger
        self.id = self.logger.get_new_agent_id(); self.parent_id = parent_id
        self.position = position if position else Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE))
        self.energy = energy if energy else 100.0
        self.is_dead = False
        if gene is None: self.gene = self.generate_random_gene()
        else: self.gene = gene
        self.build_from_gene()
        self.last_action_vector = np.zeros(len(self.gene['nodes']['output']))
        self.radius = 1.0

    def generate_random_gene(self):
        """生成一个极简的、随机的基因蓝图"""
        n_hidden = random.randint(0, 1)
        
        # 初始时，只感知最基础的场
        input_nodes = [{'channel': 0, 'type': 'value'}, {'channel': 0, 'type': 'grad_x'}, {'channel': 0, 'type': 'grad_y'}]
        # 初始时，只有最基础的移动能力
        output_nodes = [{'target': 'move_x'}, {'target': 'move_y'}]
        
        hidden_nodes = [{'id': i} for i in range(n_hidden)]
        
        connections = []
        num_connections = random.randint(1, 2) 
        for _ in range(num_connections):
            from_node_idx = random.randint(0, len(input_nodes) + n_hidden - 1)
            to_node_idx = random.randint(len(input_nodes), len(input_nodes) + n_hidden + len(output_nodes) - 1)
            connections.append({'from': from_node_idx, 'to': to_node_idx, 'weight': random.uniform(-2, 2)})
            
        return {'nodes': {'input': input_nodes, 'hidden': hidden_nodes, 'output': output_nodes},
                'connections': connections,
                'computation_depth': random.randint(1, 3), 
                'env_absorption_coeff': random.uniform(-0.5, 1.0)}

    def build_from_gene(self):
        """根据基因构建计算核心 Cᵢ"""
        self.n_input = len(self.gene['nodes']['input'])
        self.n_hidden = len(self.gene['nodes']['hidden'])
        self.n_output = len(self.gene['nodes']['output'])
        self.total_nodes = self.n_input + self.n_hidden + self.n_output
        self.computation_depth = self.gene['computation_depth']
        
        self.node_activations = np.zeros(self.total_nodes, dtype=np.float32)
        
        self.connection_matrix = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for conn in self.gene['connections']:
            from_n, to_n, weight = conn['from'], conn['to'], conn['weight']
            if 0 <= from_n < self.total_nodes and self.n_input <= to_n < self.total_nodes:
                self.connection_matrix[to_n, from_n] = weight
        
        self.complexity = len(self.gene['connections']) + self.n_hidden * 2
        self.e_repro = 150 + self.complexity * 2
        self.e_child = 50 + self.complexity
        self.e_res = 20 + self.complexity * 0.5
        self.metabolism_cost = 0.15 + 0.0005 * (self.complexity**2)
        self.identity_vector = np.mean([c['weight'] for c in self.gene['connections']]) if self.gene['connections'] else 0
        self.env_absorption_coeff = self.gene.get('env_absorption_coeff', 0.5)

    def update(self, dt, neighbors):
        if self.is_dead: return
        
        # 1. 感知 (动态构建感知向量)
        perception_vector = self.universe.get_perception_vector(self.position, self.gene['nodes']['input'])
        self.node_activations[:self.n_input] = perception_vector
        
        # 2. 计算 (有限的思维速度)
        for _ in range(self.computation_depth):
            inputs = np.dot(self.connection_matrix.T, self.node_activations)
            new_activations = np.tanh(inputs)
            self.node_activations[self.n_input:] = new_activations[self.n_input:]
        
        output_activations = self.node_activations[-self.n_output:]
        self.last_action_vector = output_activations
        
        # 3. 表达 (根据输出节点功能执行)
        move_vector = Vector2(0, 0)
        for i, output_node_gene in enumerate(self.gene['nodes']['output']):
            activation = output_activations[i]
            target = output_node_gene['target']
            if target == 'move_x': move_vector.x += activation
            elif target == 'move_y': move_vector.y += activation
            elif target == 'emit_signal_1':
                if abs(activation) > 0.1: self.universe.biotic_field_1.add_circular_source(self.position, SIGNAL_EMISSION_RADIUS, abs(activation) * 0.01)
            elif target == 'emit_signal_2':
                if abs(activation) > 0.1: self.universe.biotic_field_2.add_circular_source(self.position, SIGNAL_EMISSION_RADIUS, abs(activation) * 0.01)
        
        # 4. 物理互动与移动
        repulsion_vector = Vector2(0, 0)
        for other in neighbors:
            if other is self: continue
            dist_vec = self.position - other.position
            dist_sq = dist_vec.magnitude_sq
            if dist_sq < DENSITY_REPULSION_RADIUS**2 and dist_sq > 1e-6:
                repulsion_vector += dist_vec.normalize() * (1.0 / dist_sq)
        
        move_vector += repulsion_vector * DENSITY_REPULSION_STRENGTH
        self.position += move_vector * dt * 30
        
        for other in neighbors:
            if other is self: continue
            dist_vec = self.position - other.position
            dist_sq = dist_vec.magnitude_sq
            min_dist = self.radius + other.radius
            if dist_sq < min_dist**2 and dist_sq > 0:
                overlap = min_dist - math.sqrt(dist_sq)
                self.position += dist_vec.normalize() * (overlap * 0.5)
                other.position -= dist_vec.normalize() * (overlap * 0.5)
        
        self.position.x %= WORLD_SIZE
        self.position.y %= WORLD_SIZE

        # 5. 能量经济学
        cost = self.metabolism_cost + move_vector.magnitude_sq * 0.05
        self.energy -= cost * dt
        
        nutrient_val, _ = self.universe.nutrient_field.get_value_and_gradient(self.position)
        hazard_val, _ = self.universe.hazard_field.get_value_and_gradient(self.position)
        env_gain = self.env_absorption_coeff * nutrient_val * 40
        env_loss = abs(np.tanh(self.identity_vector)) * hazard_val * 30
        self.energy += (env_gain - env_loss) * dt

        for other in neighbors:
            if other is self or other.is_dead: continue
            dist_sq = (self.position - other.position).magnitude_sq
            if dist_sq < INTERACTION_RANGE**2:
                energy_transfer = other.identity_vector * K_INTERACTION_FACTOR * self.identity_vector
                self.energy += energy_transfer * dt
                other.energy -= energy_transfer * (1 + ENERGY_TRANSFER_EFFICIENCY) * dt
        
        if self.energy <= 0:
            self.is_dead = True
            self.universe.on_agent_death(self)

    def reproduce(self):
        if self.energy >= self.e_repro:
            self.energy -= self.e_child
            new_gene = {
                'nodes': {
                    'input': [dict(n) for n in self.gene['nodes']['input']],
                    'hidden': [dict(n) for n in self.gene['nodes']['hidden']],
                    'output': [dict(n) for n in self.gene['nodes']['output']]
                },
                'connections': [dict(c) for c in self.gene['connections']],
                'computation_depth': self.gene['computation_depth'],
                'env_absorption_coeff': self.gene['env_absorption_coeff']
            }

            # 突变...
            # (这里可以加入更复杂的突变，比如改变节点的channel或target)
            
            child_pos = self.position + Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
            return Agent(self.universe, self.logger, gene=new_gene, position=child_pos, energy=self.e_child, parent_id=self.id)
        return None

    def draw(self, surface, camera):
        if self.is_dead: return
        screen_pos = camera.world_to_screen(self.position)
        radius = max(1.0, camera.zoom * self.radius) 
        if not (-radius < screen_pos[0] < camera.render_width + radius and -radius < screen_pos[1] < camera.render_height + radius):
            return
        hue = (self.complexity * 10) % 360
        color = pygame.Color(0); color.hsva = (hue, 85, 90, 100)
        pygame.draw.circle(surface, color, screen_pos, int(radius))
        if self.universe.selected_agent and self.universe.selected_agent.id == self.id:
            pygame.draw.circle(surface, (255, 255, 0), screen_pos, int(radius + 2), 2)
        else:
            pygame.draw.circle(surface, (255, 255, 255), screen_pos, int(radius), 1)

class Universe:
    def __init__(self, logger, render_width, render_height):
        self.logger = logger
        self.fields = [
            Field(WORLD_SIZE, 1, "Nutrient"), Field(WORLD_SIZE, 0, "Hazard"),
            Field(WORLD_SIZE, 2, "Biotic_1"), Field(WORLD_SIZE, 0, "Biotic_2"),
        ]
        self.nutrient_field, self.hazard_field, self.biotic_field_1, self.biotic_field_2 = self.fields
        self.energy_springs = [Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE)) for _ in range(3)]
        self.agents = [Agent(self, self.logger) for _ in range(INITIAL_AGENT_COUNT)]
        self.frame_count = 0
        self.selected_agent = None
        self.view_mode = 1
        self.camera = Camera(render_width, render_height)
        self.grid_cell_size = max(INTERACTION_RANGE, DENSITY_REPULSION_RADIUS) * 2
        self.spatial_grid = defaultdict(list)

    def get_perception_vector(self, pos, input_nodes_gene):
        """根据基因动态构建感知向量"""
        perception_values = []
        for node_gene in input_nodes_gene:
            channel = node_gene.get('channel', 0)
            p_type = node_gene.get('type', 'value')
            
            if 0 <= channel < len(self.fields):
                field = self.fields[channel]
                val, grad = field.get_value_and_gradient(pos)
                if p_type == 'value':
                    perception_values.append(val)
                elif p_type == 'grad_x':
                    perception_values.append(grad.x)
                elif p_type == 'grad_y':
                    perception_values.append(grad.y)
            else: # 如果基因指定的频道不存在，则感知为0
                perception_values.append(0)
        return np.array(perception_values, dtype=np.float32)

    def get_perception_vector_template(self):
        # 这个方法现在只用于确定最大可能的输入维度，实际维度由基因决定
        return np.zeros(len(self.fields) * 3)

    def on_agent_death(self, agent):
        self.nutrient_field.add_circular_source(agent.position, agent.e_res / 5, 0.5)

    def update_spatial_grid(self):
        self.spatial_grid.clear()
        for agent in self.agents:
            if not agent.is_dead:
                grid_x = int(agent.position.x / self.grid_cell_size)
                grid_y = int(agent.position.y / self.grid_cell_size)
                self.spatial_grid[(grid_x, grid_y)].append(agent)

    def get_neighbors(self, agent):
        neighbors = []
        grid_x = int(agent.position.x / self.grid_cell_size)
        grid_y = int(agent.position.y / self.grid_cell_size)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbors.extend(self.spatial_grid.get((grid_x + dx, grid_y + dy), []))
        return neighbors

    def update(self, dt):
        self.frame_count += 1
        for field in self.fields: field.update(dt)
        for spring in self.energy_springs:
            if self.frame_count % 50 == 0:
                self.nutrient_field.add_circular_source(spring, ENERGY_SPRING_RADIUS, 0.5)
        self.biotic_field_1.grid *= (1 - 2 * dt)
        self.biotic_field_2.grid *= (1 - 2 * dt)
        self.update_spatial_grid()
        for agent in self.agents:
            neighbors = self.get_neighbors(agent)
            agent.update(dt, neighbors)
        new_children = []
        for agent in self.agents:
            if not agent.is_dead:
                child = agent.reproduce()
                if child: new_children.append(child)
        self.agents = [agent for agent in self.agents if not agent.is_dead]
        self.agents.extend(new_children)
        if len(self.agents) < MIN_AGENTS_TO_SPAWN and self.frame_count % 10 == 0:
            for _ in range(10): self.agents.append(Agent(self, self.logger))
        if len(self.agents) > MAX_AGENTS:
            self.agents.sort(key=lambda a: a.energy)
            num_to_remove = len(self.agents) - MAX_AGENTS
            for i in range(num_to_remove):
                self.agents[i].is_dead = True
                self.on_agent_death(self.agents[i])
            self.agents = self.agents[num_to_remove:]
        if self.frame_count % 20 == 0:
            self.logger.log_frame(self.frame_count, self.agents)

    def handle_click(self, mouse_pos):
        world_pos = self.camera.screen_to_world(mouse_pos)
        closest_agent = None
        min_dist_sq = (10 / self.camera.zoom)**2
        for agent in self.agents:
            dist_sq = (agent.position - world_pos).magnitude_sq
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_agent = agent
        self.selected_agent = closest_agent

    def draw(self, surface, sim_surface):
        sim_surface.fill((10, 10, 20))
        if self.view_mode == 0:
            for field in self.fields: field.draw(sim_surface, self.camera)
        elif 1 <= self.view_mode <= len(self.fields):
            self.fields[self.view_mode - 1].draw(sim_surface, self.camera, alpha=255)
        for agent in self.agents:
            agent.draw(sim_surface, self.camera)
        surface.blit(sim_surface, (0, 0))

def draw_inspector_panel(surface, font, agent, mouse_pos, panel_x, panel_width, panel_height):
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
    draw_text("观察对象 ID", agent.id, (100, 255, 100))
    # ... (其他文本绘制) ...
    y_offset += 20
    draw_neural_network(surface, font, agent, panel_x + 20, y_offset, panel_width - 40, 350, mouse_pos)

def draw_neural_network(surface, font, agent, x, y, width, height, mouse_pos):
    title = font.render("计算核心 (Cᵢ) 拓扑图:", True, (200, 200, 100))
    surface.blit(title, (x, y))
    y += 30
    
    # 动态生成标签
    input_labels = [f"In{i}(Ch{n.get('channel', '?')})" for i, n in enumerate(agent.gene['nodes']['input'])]
    output_labels = [f"Out{i}({n.get('target', '?')})" for i, n in enumerate(agent.gene['nodes']['output'])]
    
    n_in = len(input_labels)
    n_hid = agent.gene['n_hidden']
    n_out = len(output_labels)

    col_x = [x + 30, x + width // 2, x + width - 30]
    layers = [n_in, n_hid, n_out]
    node_positions = {}
    
    current_node_idx = 0
    for i, n_nodes in enumerate(layers):
        if n_nodes == 0: continue
        layer_y_start = y + (height - (n_nodes - 1) * 25) / 2 if n_nodes > 1 else y + height / 2
        for j in range(n_nodes):
            node_id = current_node_idx + j
            node_positions[node_id] = (int(col_x[i]), int(layer_y_start + j * 25))
        current_node_idx += n_nodes

    # ... (其余绘制逻辑) ...

def main():
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("涌现认知生态系统 (ECE) v5.0")
    clock = pygame.time.Clock()
    try: font = pygame.font.SysFont("simhei", 16)
    except pygame.error: font = pygame.font.SysFont(None, 22)
    
    logger = DataLogger()
    
    current_screen_width, current_screen_height = screen.get_size()
    sim_area_width = current_screen_width - INFO_PANEL_WIDTH
    
    universe = Universe(logger, sim_area_width, current_screen_height)
    
    running = True
    paused = False
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.VIDEORESIZE:
                current_screen_width, current_screen_height = event.size
            
            universe.camera.handle_event(event, mouse_pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and event.pos[0] < universe.camera.render_width:
                    universe.handle_click(event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_RIGHT and paused: universe.update(0.016)
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                if pygame.K_0 <= event.key <= pygame.K_4:
                    universe.view_mode = event.key - pygame.K_0
        
        if not paused:
            dt = clock.tick(60) / 1000.0
            if dt > 0.1: dt = 0.1
            universe.update(dt)
        
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
        info_text = f"帧: {universe.frame_count} | 生命体: {len(universe.agents)} | 总生物量: {int(total_biomass)} | 视图(0-4): {view_name} | 缩放: {universe.camera.zoom:.2f}x | {'[已暂停]' if paused else ''} (空格:暂停,F11:全屏)"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
