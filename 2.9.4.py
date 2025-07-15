# -*- coding: utf-8 -*-
# =============================================================================
# 涌现认知生态系统 (ECE) - v4.3 (封闭生态版)
#
# 作者: 一个中国的高中复读生 & Gemini
# 日期: 2025年7月15日
#
# v4.3 核心功能更新:
# 1. [新增] 严格碰撞物理：智能体之间不再重叠，实现为硬球模型，增强了物理真实性。
# 2. [新增] 封闭能量系统：移除了周期性能量补给。所有能量均来自模拟开始时的一次性投放，
#    生态系统必须依赖死亡智能体的能量回收才能延续。
# 3. [调整] 智能体半径增大，使其在视觉上更清晰，碰撞效果更明显。
# =============================================================================

import pygame
import numpy as np
import random
import math
import os
import datetime
import csv
import json
from collections import defaultdict
from pygame.math import Vector2

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
DENSITY_REPULSION_RADIUS = 15.0 # 稍微增大了排斥范围
DENSITY_REPULSION_STRENGTH = 2.5
SIGNAL_EMISSION_RADIUS = 20.0 
BIOTIC_FIELD_SPECIAL_DECAY = 2.0
AGENT_RADIUS = 2.0 # [调整] 增大了智能体的基础半径

# --- 数据日志系统 (增强版) ---
class DataLogger:
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("logs", f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.state_log_path = os.path.join(self.log_dir, "simulation_log.csv")
        self.state_header = ["frame", "agent_id", "parent_id", "genotype_id", "is_mutant", "energy", "pos_x", "pos_y", "n_hidden", "n_connections", "computation_depth", "gene_string"]
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
                row = [frame_number, agent.id, agent.parent_id, agent.genotype_id, agent.is_mutant, round(agent.energy, 2), round(agent.position.x, 2), round(agent.position.y, 2), agent.gene['n_hidden'], len(agent.gene['connections']), agent.gene['computation_depth'], gene_str]
                writer.writerow(row)

    def log_event(self, frame, event_type, details):
        with open(self.event_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            details_str = json.dumps(details)
            writer.writerow([frame, event_type, details_str])

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
        full_x_coords, full_y_coords = np.meshgrid(x_range, y_range)
        
        mask = full_x_coords**2 + full_y_coords**2 <= radius**2
        
        x_coords_masked = full_x_coords[mask]
        y_coords_masked = full_y_coords[mask]
        
        distances = np.sqrt(x_coords_masked**2 + y_coords_masked**2)
        gradient_values = value * np.maximum(0, 1 - distances / radius)

        x_indices_abs = (x_center + x_coords_masked) % self.size
        y_indices_abs = (y_center + y_coords_masked) % self.size
        
        current_values_at_target = self.grid[y_indices_abs, x_indices_abs]
        room_to_add = np.maximum(0, 1.0 - current_values_at_target)
        values_to_add = np.minimum(gradient_values, room_to_add)

        self.grid[y_indices_abs, x_indices_abs] += values_to_add

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
    def __init__(self, universe, logger, gene=None, position=None, energy=None, parent_id=None, is_mutant=False):
        self.universe = universe; self.logger = logger
        self.id = self.logger.get_new_agent_id(); self.parent_id = parent_id
        self.position = position if position else Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE))
        self.energy = energy if energy else 100.0
        self.is_dead = False
        self.is_mutant = is_mutant
        self.genotype_id = None
        if gene is None: self.gene = self.generate_random_gene()
        else: self.gene = gene
        self.build_from_gene()
        self.last_action_vector = np.zeros(self.gene['n_output'])
        self.radius = AGENT_RADIUS # [调整] 使用常量设置半径

    def generate_random_gene(self):
        n_input = len(self.universe.get_perception_vector_template())
        n_output = 4
        n_hidden = random.randint(0, 1)
        connections = []
        num_connections = random.randint(1, 3) 
        for _ in range(num_connections):
            from_node = random.randint(0, n_input - 1)
            to_node = random.randint(n_input + n_hidden, n_input + n_hidden + n_output - 1)
            connections.append([from_node, to_node, random.uniform(-2, 2)])
        return {'n_input': n_input, 'n_output': n_output, 'n_hidden': n_hidden, 
                'computation_depth': random.randint(1, 3), 
                'connections': connections,
                'env_absorption_coeff': random.uniform(-0.5, 1.0)}

    def build_from_gene(self):
        self.n_input = self.gene['n_input']
        self.n_output = self.gene['n_output']
        self.n_hidden = self.gene['n_hidden']
        self.total_nodes = self.n_input + self.n_hidden + self.n_output
        self.computation_depth = self.gene['computation_depth']
        self.node_activations = np.zeros(self.total_nodes, dtype=np.float32)
        self.connection_matrix = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for from_n, to_n, weight in self.gene['connections']:
            if 0 <= from_n < self.total_nodes and self.n_input <= to_n < self.total_nodes:
                self.connection_matrix[to_n, from_n] = weight
        self.complexity = len(self.gene['connections']) + self.n_hidden * 2
        self.e_repro = 150 + self.complexity * 2
        self.e_child = 50 + self.complexity
        self.e_res = 20 + self.complexity * 0.5
        self.metabolism_cost = 0.15 + 0.0005 * (self.complexity**2)
        self.identity_vector = np.mean([c[2] for c in self.gene['connections']]) if self.gene['connections'] else 0
        self.env_absorption_coeff = self.gene.get('env_absorption_coeff', 0.5)
        self.genotype_id = self.universe.get_or_create_genotype_id(self.gene)

    def update(self, dt, neighbors):
        if self.is_dead: return
        
        # 1. 感知与决策
        perception_vector = self.universe.get_perception_vector(self.position)
        self.node_activations[:self.n_input] = perception_vector
        for _ in range(self.computation_depth):
            inputs = np.dot(self.connection_matrix.T, self.node_activations)
            new_activations = np.tanh(inputs)
            self.node_activations[self.n_input:] = new_activations[self.n_input:]
        output_activations = self.node_activations[-self.n_output:]
        self.last_action_vector = output_activations
        
        # 2. 计算行为向量 (包括密度排斥)
        move_vector = Vector2(output_activations[0], output_activations[1])
        repulsion_vector = Vector2(0, 0)
        close_neighbors_count = 0
        
        for other in neighbors:
            if other is self: continue
            dist_vec = self.position - other.position
            dist_sq = dist_vec.length_squared()
            if dist_sq < DENSITY_REPULSION_RADIUS**2:
                close_neighbors_count += 1
                if dist_sq > 1e-6:
                    repulsion_vector += dist_vec.normalize() / dist_sq
        
        if close_neighbors_count > 3:
            move_vector += repulsion_vector * DENSITY_REPULSION_STRENGTH
        
        # 3. 移动
        self.position += move_vector * dt * 30

        # 4. [新增] 严格碰撞解决
        for other in neighbors:
            if other is self: continue
            dist_vec = self.position - other.position
            dist_sq = dist_vec.length_squared()
            min_dist = self.radius + other.radius
            if dist_sq < min_dist**2 and dist_sq > 0:
                overlap = min_dist - math.sqrt(dist_sq)
                # 将当前智能体沿碰撞向量推开整个重叠距离
                self.position += dist_vec.normalize() * overlap

        # 5. 世界边界环绕
        self.position.x %= WORLD_SIZE
        self.position.y %= WORLD_SIZE

        # 6. 信号释放与能量交换
        signal_1_strength = abs(output_activations[2])
        signal_2_strength = abs(output_activations[3])
        if signal_1_strength > 0.1: self.universe.biotic_field_1.add_circular_source(self.position, SIGNAL_EMISSION_RADIUS, signal_1_strength * 0.01)
        if signal_2_strength > 0.1: self.universe.biotic_field_2.add_circular_source(self.position, SIGNAL_EMISSION_RADIUS, signal_2_strength * 0.01)

        for other in neighbors:
            if other is self or other.is_dead: continue
            dist_sq = (self.position - other.position).length_squared()
            if dist_sq < INTERACTION_RANGE**2:
                energy_transfer = other.identity_vector * K_INTERACTION_FACTOR * self.identity_vector
                self.energy += energy_transfer * dt
                other.energy -= energy_transfer * (1 - ENERGY_TRANSFER_EFFICIENCY) * dt

        # 7. 新陈代谢与环境能量吸收
        cost = self.metabolism_cost + move_vector.length_squared() * 0.05 + (signal_1_strength + signal_2_strength) * 0.2
        nutrient_val, _ = self.universe.nutrient_field.get_value_and_gradient(self.position)
        hazard_val, _ = self.universe.hazard_field.get_value_and_gradient(self.position)
        env_gain = self.env_absorption_coeff * nutrient_val * 40
        env_loss = abs(np.tanh(self.identity_vector)) * hazard_val * 30
        self.energy += (env_gain - env_loss - cost) * dt

        # 8. 死亡判定
        if self.energy <= 0:
            self.is_dead = True
            self.logger.log_event(self.universe.frame_count, 'AGENT_DEATH', {'agent_id': self.id, 'reason': 'energy_depleted'})
            self.universe.on_agent_death(self)

    def reproduce(self):
        if self.energy < self.e_repro:
            return None

        self.energy -= self.e_child
        new_gene = json.loads(json.dumps(self.gene))
        mutations_occurred = []

        for conn in new_gene['connections']:
            if random.random() < MUTATION_PROBABILITY['point']:
                conn[2] += random.uniform(-1, 1) * MUTATION_STRENGTH
                mutations_occurred.append('point_mutation')
        if random.random() < MUTATION_PROBABILITY['add_conn']:
            from_n = random.randint(0, self.n_input + self.n_hidden - 1)
            to_n = random.randint(self.n_input, self.total_nodes - 1)
            new_gene['connections'].append([from_n, to_n, random.uniform(-1, 1)])
            mutations_occurred.append('add_connection')
        if len(new_gene['connections']) > self.n_input and random.random() < MUTATION_PROBABILITY['del_conn']:
            new_gene['connections'].pop(random.randrange(len(new_gene['connections'])))
            mutations_occurred.append('delete_connection')
        if 'env_absorption_coeff' in new_gene and random.random() < MUTATION_PROBABILITY['point']:
            new_gene['env_absorption_coeff'] += random.uniform(-1, 1) * MUTATION_STRENGTH
            mutations_occurred.append('absorption_coeff_mutation')

        is_mutant_child = len(mutations_occurred) > 0
        child_pos = self.position + Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
        child = Agent(self.universe, self.logger, gene=new_gene, position=child_pos, energy=self.e_child, parent_id=self.id, is_mutant=is_mutant_child)
        
        if is_mutant_child:
            self.logger.log_event(
                self.universe.frame_count, 
                'MUTATION', 
                {'parent_id': self.id, 'parent_genotype': self.genotype_id, 'child_id': child.id, 'child_genotype': child.genotype_id, 'types': list(set(mutations_occurred))}
            )
        return child

    def draw(self, surface, camera):
        if self.is_dead: return
        screen_pos = camera.world_to_screen(self.position)
        radius = max(1.0, camera.zoom * self.radius) 
        
        if not (-radius < screen_pos[0] < camera.render_width + radius and -radius < screen_pos[1] < camera.render_height + radius):
            return
        
        hue = (self.genotype_id * 20) % 360
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
            Field(WORLD_SIZE, 1, "Nutrient/Energy"), Field(WORLD_SIZE, 0, "Hazard"),
            Field(WORLD_SIZE, 2, "Biotic 1"), Field(WORLD_SIZE, 0, "Biotic 2"),
        ]
        self.nutrient_field, self.hazard_field, self.biotic_field_1, self.biotic_field_2 = self.fields
        self.frame_count = 0
        self.selected_agent = None
        self.view_mode = 1
        self.camera = Camera(render_width, render_height)
        self.grid_cell_size = INTERACTION_RANGE * 1.1
        self.spatial_grid = defaultdict(list)
        
        self.genotype_registry = {}
        self.next_genotype_id = 0
        
        # [新增] 封闭能量系统：在模拟开始时一次性投放能量
        self._initial_energy_seeding()
        
        self.agents = [Agent(self, self.logger) for _ in range(INITIAL_AGENT_COUNT)]

    def _initial_energy_seeding(self):
        """在世界中一次性播种初始能量。"""
        num_patches = 15
        for _ in range(num_patches):
            pos = Vector2(random.uniform(0, WORLD_SIZE), random.uniform(0, WORLD_SIZE))
            radius = random.uniform(30, 60)
            self.nutrient_field.add_circular_source(pos, radius, 1.0)
        self.logger.log_event(0, 'INITIAL_ENERGY_SEED', {'patches': num_patches})

    def _get_canonical_gene(self, gene):
        """将基因字典转换为可哈希的、唯一的表示形式"""
        sorted_connections = tuple(sorted(tuple(c) for c in gene['connections']))
        canonical_items = tuple(sorted((k, v if k != 'connections' else sorted_connections) for k, v in gene.items()))
        return canonical_items

    def get_or_create_genotype_id(self, gene):
        """获取或创建一个新的基因型ID"""
        canonical_gene = self._get_canonical_gene(gene)
        if canonical_gene not in self.genotype_registry:
            self.genotype_registry[canonical_gene] = self.next_genotype_id
            self.next_genotype_id += 1
        return self.genotype_registry[canonical_gene]

    def get_perception_vector(self, pos):
        perception = []
        for field in self.fields:
            val, grad = field.get_value_and_gradient(pos)
            perception.extend([val, grad.x, grad.y])
        return np.array(perception, dtype=np.float32)
    
    def get_perception_vector_template(self): return np.zeros(len(self.fields) * 3)
    def on_agent_death(self, agent): self.nutrient_field.add_circular_source(agent.position, agent.e_res / 4, 0.8)

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
        grid_w = int(WORLD_SIZE / self.grid_cell_size)
        grid_h = int(WORLD_SIZE / self.grid_cell_size)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                wrapped_x = (grid_x + dx) % grid_w
                wrapped_y = (grid_y + dy) % grid_h
                neighbors.extend(self.spatial_grid.get((wrapped_x, wrapped_y), []))
        return neighbors

    def update(self, dt):
        self.frame_count += 1
        for field in self.fields: field.update(dt)
        
        # [移除] 周期性能量补给，实现封闭生态
        # for spring in self.energy_springs:
        #     if self.frame_count % 50 == 0:
        #         self.nutrient_field.add_circular_source(spring, ENERGY_SPRING_RADIUS, 0.5)
        
        self.biotic_field_1.grid *= (1 - BIOTIC_FIELD_SPECIAL_DECAY * dt)
        self.biotic_field_2.grid *= (1 - BIOTIC_FIELD_SPECIAL_DECAY * dt)

        self.update_spatial_grid()
        for agent in self.agents:
            agent.update(dt, self.get_neighbors(agent))
        
        new_children = []
        for agent in list(self.agents):
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
            culled_ids = [a.id for a in self.agents[:num_to_remove]]
            for agent_to_remove in self.agents[:num_to_remove]:
                agent_to_remove.is_dead = True
                self.on_agent_death(agent_to_remove)
            self.agents = self.agents[num_to_remove:]
            self.logger.log_event(self.frame_count, 'CULL', {'count': num_to_remove, 'culled_ids': culled_ids})
            
        if self.frame_count % 20 == 0:
            self.logger.log_state(self.frame_count, self.agents)

    def handle_click(self, mouse_pos):
        world_pos = self.camera.screen_to_world(mouse_pos)
        closest_agent = None
        min_dist_sq = (10 / self.camera.zoom)**2
        for agent in self.agents:
            dist_sq = (agent.position - world_pos).length_squared()
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
    
    if agent.is_mutant:
        draw_text("观察对象 ID", f"{agent.id} (M)", (255, 255, 100))
    else:
        draw_text("观察对象 ID", agent.id, (100, 255, 100))

    draw_text("亲代 ID", agent.parent_id if agent.parent_id else "N/A")
    draw_text("基因型 ID", agent.genotype_id)
    draw_text("能量 (E)", f"{agent.energy:.2f}")
    draw_text("位置 (p)", f"({agent.position.x:.1f}, {agent.position.y:.1f})")
    y_offset += 10
    draw_text("--- 基因特性 ---", "", (200, 200, 100))
    draw_text("复杂度 (Ω)", agent.complexity)
    draw_text("隐藏节点数", agent.n_hidden)
    draw_text("连接数", len(agent.gene['connections']))
    draw_text("思维深度 (k)", agent.computation_depth)
    draw_text("环境吸收系数", f"{agent.env_absorption_coeff:.2f}")
    y_offset += 10
    draw_text("--- 行为输出 ---", "", (200, 200, 100))
    draw_text("移动 X", f"{agent.last_action_vector[0]:.2f}")
    draw_text("移动 Y", f"{agent.last_action_vector[1]:.2f}")
    draw_text("信号1强度", f"{abs(agent.last_action_vector[2]):.2f}")
    draw_text("信号2强度", f"{abs(agent.last_action_vector[3]):.2f}")
    y_offset += 20
    draw_neural_network(surface, font, agent, panel_x + 20, y_offset, panel_width - 40, 350, mouse_pos)

def draw_neural_network(surface, font, agent, x, y, width, height, mouse_pos):
    title = font.render("计算核心 (Cᵢ) 拓扑图:", True, (200, 200, 100))
    surface.blit(title, (x, y))
    y += 30
    n_in, n_hid, n_out = agent.n_input, agent.n_hidden, agent.n_output
    input_labels = ["N_v", "N_gx", "N_gy", "H_v", "H_gx", "H_gy", "B1_v", "B1_gx", "B1_gy", "B2_v", "B2_gx", "B2_gy"]
    output_labels = ["MoveX", "MoveY", "Signal1", "Signal2"]
    col_x = [x + 30, x + width // 2, x + width - 30]
    layers = [n_in, n_hid, n_out]
    node_positions = {}
    
    col_map = [0, 1, 2] if n_hid > 0 else [0, 2] 
    
    current_node_idx = 0
    visible_layer_idx = 0 
    for i, n_nodes in enumerate(layers):
        if n_nodes == 0: 
            continue
        
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
            color = (0, 200, 0, 180) if weight > 0 else (255, 80, 80, 180)
            pygame.draw.aaline(surface, color, start_pos, end_pos)
    
    hover_info = None
    for node_id, pos in node_positions.items():
        is_input = node_id < n_in
        is_hidden = n_in <= node_id < n_in + n_hid
        is_output = node_id >= n_in + n_hid
        
        color = (100, 100, 255) if is_input else (255, 165, 0) if is_hidden else (255, 255, 100)
        pygame.draw.circle(surface, color, pos, 6)
        pygame.draw.circle(surface, (0,0,0), pos, 6, 1)

        label = None
        if is_input and node_id < len(input_labels):
            label = input_labels[node_id]
        elif is_output:
            output_idx = node_id - (n_in + n_hid)
            if output_idx < len(output_labels):
                label = output_labels[output_idx]

        if label:
            label_surf = font.render(label, True, (200, 200, 200))
            if is_input:
                surface.blit(label_surf, (pos[0] - label_surf.get_width() - 5, pos[1] - 8))
            else:
                surface.blit(label_surf, (pos[0] + 10, pos[1] - 8))
        
        if math.hypot(mouse_pos[0] - pos[0], mouse_pos[1] - pos[1]) < 6:
            hover_info = (f"Node {node_id}", f"Activation: {agent.node_activations[node_id]:.3f}", mouse_pos)

    if hover_info:
        title, value, pos = hover_info
        title_surf = font.render(title, True, (255, 255, 255))
        value_surf = font.render(value, True, (255, 255, 255))
        box_rect = pygame.Rect(pos[0] + 10, pos[1] + 10, max(title_surf.get_width(), value_surf.get_width()) + 20, 50)
        pygame.draw.rect(surface, (0,0,0,200), box_rect)
        surface.blit(title_surf, (box_rect.x + 10, box_rect.y + 5))
        surface.blit(value_surf, (box_rect.x + 10, box_rect.y + 25))

def main():
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("涌现认知生态系统 (ECE) v4.3 - 封闭生态版")
    clock = pygame.time.Clock()
    try: font = pygame.font.SysFont("simhei", 16)
    except pygame.error: font = pygame.font.SysFont(None, 22)
    
    logger = DataLogger()
    
    current_screen_width, current_screen_height = screen.get_size()
    sim_area_width = current_screen_width - INFO_PANEL_WIDTH
    
    universe = Universe(logger, sim_area_width, current_screen_height)
    
    logger.log_event(0, 'SIM_START', {'initial_agents': INITIAL_AGENT_COUNT, 'world_size': WORLD_SIZE})
    
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
                if event.button == 1 and mouse_pos[0] < universe.camera.render_width:
                    universe.handle_click(event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_RIGHT and paused: universe.update(0.016)
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                if pygame.K_0 <= event.key <= pygame.K_4:
                    universe.view_mode = event.key - pygame.K_0
        
        if not paused:
            dt = min(clock.tick(60) / 1000.0, 0.1)
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
        info_text = f"帧: {universe.frame_count} | 生命体: {len(universe.agents)} ({universe.next_genotype_id}个基因型) | 总生物量: {int(total_biomass)} | 视图(0-4): {view_name} | {'[已暂停]' if paused else ''}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
