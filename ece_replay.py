#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# 涌现认知生态系统 (ECE) - 回放器
# 用于播放ECE生成的CSV日志文件，进行后续观察和分析
# =============================================================================

import pygame
import numpy as np
import os
import sys
import csv
import json
import datetime
import tkinter as tk
from tkinter import filedialog
from pygame.math import Vector2
import pandas as pd
import time

# --- 常量设置 ---
INITIAL_SCREEN_WIDTH = 1200
INITIAL_SCREEN_HEIGHT = 800
WORLD_SIZE = 512  # 保持与原模拟相同的世界大小
INFO_PANEL_WIDTH = 400
FRAME_RATE = 60

# --- 界面颜色 ---
BACKGROUND_COLOR = (10, 10, 20)
PANEL_COLOR = (40, 40, 60, 220)
TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 200, 100)
EVENT_COLORS = {
    'SIM_START': (100, 255, 100),
    'AGENT_DEATH': (255, 100, 100),
    'MUTATION': (255, 255, 100),
    'PREDATION': (255, 150, 100),
    'SPAWN_NEW': (100, 200, 255),
    'SPAWN_WARNING': (255, 200, 100),
    'CULL': (200, 100, 255)
}

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

# --- 回放智能体 ---
class ReplayAgent:
    def __init__(self, agent_data):
        self.id = agent_data['agent_id']
        self.parent_id = agent_data['parent_id']
        self.genotype_id = agent_data['genotype_id']
        self.is_mutant = agent_data['is_mutant']
        self.energy = agent_data['energy']
        self.position = Vector2(agent_data['pos_x'], agent_data['pos_y'])
        self.n_hidden = agent_data['n_hidden']
        self.n_connections = agent_data['n_connections']
        self.computation_depth = agent_data['computation_depth']
        try:
            self.gene = json.loads(agent_data['gene_string'].replace("'", '"'))
        except:
            self.gene = {"connections": []}
        self.radius = 2.0  # 保持与原模拟相同的半径
        self.is_selected = False
        
    def draw(self, surface, camera):
        # 检查智能体是否在视口内
        screen_pos = camera.world_to_screen(self.position)
        if (screen_pos[0] < -50 or screen_pos[0] > camera.render_width + 50 or
            screen_pos[1] < -50 or screen_pos[1] > camera.render_height + 50):
            return  # 如果不在视口内，不绘制
        
        # 计算屏幕上的半径
        radius = max(1, int(self.radius * camera.zoom))
        
        # 基于基因型ID设置颜色 - 与原模拟一致
        hue = (self.genotype_id * 20) % 360
        color = pygame.Color(0)
        color.hsva = (hue, 85, 90, 100)
        
        # 根据能量水平调整亮度
        energy_ratio = min(1.0, self.energy / 20)  # 假设繁殖阈值为20
        if energy_ratio < 0.3:
            # 能量不足时颜色变暗
            _, s, v, _ = color.hsva
            color.hsva = (hue, s, max(30, int(v * energy_ratio / 0.3)), 100)
            
        # 绘制智能体主体
        pygame.draw.circle(surface, color, screen_pos, radius)
        
        # 如果是被选中的智能体，绘制选中标记
        if self.is_selected:
            # 绘制选中标记
            highlight_radius = radius + 3
            pygame.draw.circle(surface, (255, 255, 255), screen_pos, highlight_radius, 1)
            
            # 绘制ID标记
            id_color = (255, 200, 100) if self.is_mutant else (100, 200, 255)
            id_text = str(self.id)
            
            font = pygame.font.SysFont(None, 14)
            text_surface = font.render(id_text, True, id_color)
            text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - radius - 10))
            surface.blit(text_surface, text_rect)
        else:
            # 非选中智能体绘制简单轮廓
            if radius <= 2:
                pygame.draw.circle(surface, color, screen_pos, 1)
            else:
                pygame.draw.circle(surface, color, screen_pos, radius, 1)

# --- 回放控制器 ---
class ReplayController:
    def __init__(self, screen_width, screen_height, font):
        self.sim_log_path = None
        self.event_log_path = None
        self.frame_data = {}  # 帧号到智能体数据的映射
        self.event_data = {}  # 帧号到事件的映射
        self.frames = []      # 所有帧号
        self.current_frame_idx = 0
        self.playing = False
        self.playback_speed = 1.0
        self.selected_agent_id = None
        self.genotype_colors = {}
        self.font = font  # 使用传入的字体
        
        # 计算界面布局
        self.update_layout(screen_width, screen_height)
        
        # 初始化相机
        self.camera = Camera(self.sim_area_width, screen_height)
        
    def update_layout(self, screen_width, screen_height):
        """更新界面布局参数"""
        # 计算模拟区域和信息面板的尺寸
        self.sim_area_width = screen_width - INFO_PANEL_WIDTH
        if self.sim_area_width < 400:  # 确保模拟区域至少有一定宽度
            self.sim_area_width = 400
        self.info_panel_width = screen_width - self.sim_area_width
        self.screen_height = screen_height
        
        # 更新控制元素位置
        self.slider_rect = pygame.Rect(50, screen_height - 30, self.sim_area_width - 100, 20)
        self.slider_button_rect = pygame.Rect(50, screen_height - 35, 10, 30)
        self.slider_dragging = False
        
        # 播放控制按钮
        button_y = screen_height - 60
        self.play_button_rect = pygame.Rect(50, button_y, 30, 20)
        self.pause_button_rect = pygame.Rect(90, button_y, 30, 20)
        self.speed_up_rect = pygame.Rect(130, button_y, 30, 20)
        self.speed_down_rect = pygame.Rect(170, button_y, 30, 20)
        
        # 如果相机已初始化，更新相机渲染尺寸
        if hasattr(self, 'camera'):
            self.camera.update_render_size(self.sim_area_width, screen_height)
        
    def select_log_file(self):
        """选择日志文件目录"""
        root = tk.Tk()
        root.withdraw()
        log_dir = filedialog.askdirectory(title="选择日志目录")
        if log_dir:
            self.sim_log_path = os.path.join(log_dir, "simulation_log.csv")
            self.event_log_path = os.path.join(log_dir, "event_log.csv")
            
            if os.path.exists(self.sim_log_path) and os.path.exists(self.event_log_path):
                self.load_data()
                return True
        
        return False
    
    def load_data(self):
        """加载日志数据"""
        print("正在加载模拟数据...")
        start_time = time.time()
        
        # 使用pandas加载数据，更高效
        sim_df = pd.read_csv(self.sim_log_path)
        event_df = pd.read_csv(self.event_log_path)
        
        # 转换为字典以便快速查询
        self.frame_data = {}
        self.event_data = {}
        
        # 处理模拟数据
        for frame in sim_df['frame'].unique():
            frame_df = sim_df[sim_df['frame'] == frame]
            self.frame_data[int(frame)] = []
            
            for _, row in frame_df.iterrows():
                agent_data = {
                    'agent_id': row['agent_id'],
                    'parent_id': row['parent_id'],
                    'genotype_id': row['genotype_id'],
                    'is_mutant': bool(row['is_mutant']),
                    'energy': row['energy'],
                    'pos_x': row['pos_x'],
                    'pos_y': row['pos_y'],
                    'n_hidden': row['n_hidden'],
                    'n_connections': row['n_connections'],
                    'computation_depth': row['computation_depth'],
                    'gene_string': row['gene_string']
                }
                self.frame_data[int(frame)].append(agent_data)
        
        # 处理事件数据
        for _, row in event_df.iterrows():
            frame = int(row['frame'])
            if frame not in self.event_data:
                self.event_data[frame] = []
                
            try:
                details = json.loads(row['details'])
            except:
                details = {}
                
            event = {
                'type': row['event_type'],
                'details': details
            }
            self.event_data[frame].append(event)
        
        # 获取所有帧号并排序
        self.frames = sorted(list(self.frame_data.keys()))
        self.current_frame_idx = 0
        
        print(f"数据加载完成，共有{len(self.frames)}帧，耗时{time.time() - start_time:.2f}秒")
    
    def get_current_frame(self):
        """获取当前帧号"""
        if not self.frames:
            return 0
        return self.frames[self.current_frame_idx]
    
    def get_current_agents(self):
        """获取当前帧的智能体数据"""
        current_frame = self.get_current_frame()
        if current_frame in self.frame_data:
            agents = []
            for agent_data in self.frame_data[current_frame]:
                agent = ReplayAgent(agent_data)
                if agent.id == self.selected_agent_id:
                    agent.is_selected = True
                agents.append(agent)
            return agents
        return []
    
    def get_current_events(self):
        """获取当前帧的事件"""
        current_frame = self.get_current_frame()
        if current_frame in self.event_data:
            return self.event_data[current_frame]
        return []
    
    def next_frame(self):
        """前进到下一帧"""
        if self.frames and self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            return True
        return False
    
    def prev_frame(self):
        """回到上一帧"""
        if self.frames and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            return True
        return False
    
    def set_frame_index(self, index):
        """设置当前帧索引"""
        if self.frames:
            self.current_frame_idx = max(0, min(index, len(self.frames) - 1))
    
    def handle_event(self, event, mouse_pos):
        """处理用户输入事件"""
        # 处理相机事件
        self.camera.handle_event(event, mouse_pos)
        
        # 处理UI控件
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键点击
                # 检查是否点击进度条
                if self.slider_rect.collidepoint(mouse_pos):
                    # 计算点击位置对应的帧索引
                    progress = (mouse_pos[0] - self.slider_rect.x) / self.slider_rect.width
                    frame_idx = int(progress * (len(self.frames) - 1))
                    self.set_frame_index(frame_idx)
                    return True
                
                # 检查是否点击滑块
                if self.slider_button_rect.collidepoint(mouse_pos):
                    self.slider_dragging = True
                    return True
                
                # 检查是否点击播放按钮
                if self.play_button_rect.collidepoint(mouse_pos):
                    self.playing = True
                    return True
                
                # 检查是否点击暂停按钮
                if self.pause_button_rect.collidepoint(mouse_pos):
                    self.playing = False
                    return True
                
                # 检查是否点击加速按钮
                if self.speed_up_rect.collidepoint(mouse_pos):
                    self.playback_speed = min(8.0, self.playback_speed * 2)
                    return True
                
                # 检查是否点击减速按钮
                if self.speed_down_rect.collidepoint(mouse_pos):
                    self.playback_speed = max(0.25, self.playback_speed / 2)
                    return True
                
                # 检查是否点击智能体
                if mouse_pos[0] < self.sim_area_width:
                    world_pos = self.camera.screen_to_world(mouse_pos)
                    closest_agent = self.find_agent_at_position(world_pos)
                    if closest_agent:
                        self.selected_agent_id = closest_agent.id
                        return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # 左键释放
                self.slider_dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            # 如果正在拖动滑块
            if self.slider_dragging:
                # 计算拖动位置对应的帧索引
                progress = (mouse_pos[0] - self.slider_rect.x) / self.slider_rect.width
                progress = max(0, min(1, progress))
                frame_idx = int(progress * (len(self.frames) - 1))
                self.set_frame_index(frame_idx)
                return True
        
        elif event.type == pygame.KEYDOWN:
            # 空格键切换播放/暂停
            if event.key == pygame.K_SPACE:
                self.playing = not self.playing
                return True
            
            # 左右箭头键控制前进/后退
            if event.key == pygame.K_RIGHT:
                self.next_frame()
                return True
            
            if event.key == pygame.K_LEFT:
                self.prev_frame()
                return True
            
            # 上下箭头键控制速度
            if event.key == pygame.K_UP:
                self.playback_speed = min(8.0, self.playback_speed * 2)
                return True
            
            if event.key == pygame.K_DOWN:
                self.playback_speed = max(0.25, self.playback_speed / 2)
                return True
            
            # F11全屏切换
            if event.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
                return True
        
        return False
    
    def find_agent_at_position(self, world_pos):
        """查找指定位置的智能体"""
        agents = self.get_current_agents()
        closest_agent = None
        min_dist_sq = (10 / self.camera.zoom)**2
        
        for agent in agents:
            dist_sq = (agent.position - world_pos).length_squared()
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_agent = agent
        
        return closest_agent
    
    def update(self, dt):
        """更新回放状态"""
        # 如果正在播放且有帧数据
        if self.playing and self.frames:
            # 根据播放速度累积时间
            self.frame_advance_time += dt * self.playback_speed
            
            # 如果累积的时间足够前进一帧
            while self.frame_advance_time >= self.frame_time:
                self.frame_advance_time -= self.frame_time
                if not self.next_frame():
                    self.playing = False  # 播放到末尾时停止
                    break
    
    def draw(self, screen):
        """绘制回放界面"""
        # 绘制背景
        screen.fill(BACKGROUND_COLOR)
        
        # 创建模拟区域的表面
        sim_surface = pygame.Surface((self.sim_area_width, self.screen_height))
        sim_surface.fill(BACKGROUND_COLOR)
        
        # 绘制智能体
        agents = self.get_current_agents()
        for agent in agents:
            agent.draw(sim_surface, self.camera)
        
        # 将模拟表面绘制到屏幕上
        screen.blit(sim_surface, (0, 0))
        
        # 绘制信息面板
        self.draw_info_panel(screen)
        
        # 绘制控制界面
        self.draw_controls(screen)
    
    def draw_controls(self, screen):
        """绘制控制界面"""
        # 绘制进度条背景
        pygame.draw.rect(screen, (100, 100, 100), self.slider_rect)
        
        # 绘制进度条填充部分
        if self.frames:
            progress = self.current_frame_idx / (len(self.frames) - 1) if len(self.frames) > 1 else 0
            fill_rect = pygame.Rect(self.slider_rect.x, self.slider_rect.y, 
                                   self.slider_rect.width * progress, self.slider_rect.height)
            pygame.draw.rect(screen, (150, 150, 250), fill_rect)
        
        # 绘制滑块
        if self.frames:
            progress = self.current_frame_idx / (len(self.frames) - 1) if len(self.frames) > 1 else 0
            self.slider_button_rect.x = self.slider_rect.x + self.slider_rect.width * progress - self.slider_button_rect.width // 2
            pygame.draw.rect(screen, (200, 200, 255), self.slider_button_rect)
        
        # 绘制播放控制按钮
        pygame.draw.rect(screen, (100, 200, 100) if self.playing else (150, 150, 150), self.play_button_rect)
        pygame.draw.rect(screen, (200, 100, 100) if not self.playing else (150, 150, 150), self.pause_button_rect)
        pygame.draw.rect(screen, (150, 150, 150), self.speed_up_rect)
        pygame.draw.rect(screen, (150, 150, 150), self.speed_down_rect)
        
        # 绘制按钮文字
        small_font = pygame.font.SysFont(None, 20)
        play_text = small_font.render("▶", True, (0, 0, 0))
        pause_text = small_font.render("❚❚", True, (0, 0, 0))
        speed_up_text = small_font.render(">>", True, (0, 0, 0))
        speed_down_text = small_font.render("<<", True, (0, 0, 0))
        
        screen.blit(play_text, (self.play_button_rect.x + 8, self.play_button_rect.y + 3))
        screen.blit(pause_text, (self.pause_button_rect.x + 8, self.pause_button_rect.y + 3))
        screen.blit(speed_up_text, (self.speed_up_rect.x + 5, self.speed_up_rect.y + 3))
        screen.blit(speed_down_text, (self.speed_down_rect.x + 5, self.speed_down_rect.y + 3))
        
        # 绘制当前帧信息和播放速度
        current_frame = self.get_current_frame()
        total_frames = self.frames[-1] if self.frames else 0
        
        frame_text = self.font.render(f"帧: {current_frame} / {total_frames}", True, TEXT_COLOR)
        speed_text = self.font.render(f"速度: {self.playback_speed:.2f}x", True, TEXT_COLOR)
        
        screen.blit(frame_text, (self.slider_rect.x + self.slider_rect.width + 20, self.slider_rect.y))
        screen.blit(speed_text, (self.slider_rect.x + self.slider_rect.width + 20, self.slider_rect.y - 20))
    
    def draw_info_panel(self, surface):
        """绘制信息面板"""
        # 创建半透明面板
        panel_surface = pygame.Surface((self.info_panel_width, self.screen_height), pygame.SRCALPHA)
        panel_surface.fill(PANEL_COLOR)
        surface.blit(panel_surface, (self.sim_area_width, 0))
        
        # 设置字体
        title_font = pygame.font.SysFont(None, 24)
        
        # 绘制标题
        title_text = self.font.render("ECE 模拟回放器", True, HIGHLIGHT_COLOR)
        surface.blit(title_text, (self.sim_area_width + 20, 20))
        
        # 绘制当前帧统计信息
        agents = self.get_current_agents()
        current_frame = self.get_current_frame()
        
        # 智能体统计
        y_offset = 60
        stats_text = self.font.render(f"帧: {current_frame} | 智能体: {len(agents)}", True, TEXT_COLOR)
        surface.blit(stats_text, (self.sim_area_width + 20, y_offset))
        y_offset += 25
        
        # 统计不同基因型数量
        genotypes = {}
        total_energy = 0
        for agent in agents:
            genotype_id = agent.genotype_id
            if genotype_id not in genotypes:
                genotypes[genotype_id] = 0
            genotypes[genotype_id] += 1
            total_energy += agent.energy
        
        genotype_text = self.font.render(f"基因型: {len(genotypes)} | 总能量: {int(total_energy)}", True, TEXT_COLOR)
        surface.blit(genotype_text, (self.sim_area_width + 20, y_offset))
        y_offset += 40
        
        # 绘制选中智能体信息
        if self.selected_agent_id is not None:
            selected_agent = None
            for agent in agents:
                if agent.id == self.selected_agent_id:
                    selected_agent = agent
                    break
            
            if selected_agent:
                agent_title = self.font.render("--- 选中智能体信息 ---", True, HIGHLIGHT_COLOR)
                surface.blit(agent_title, (self.sim_area_width + 20, y_offset))
                y_offset += 25
                
                def draw_agent_info(label, value, color=TEXT_COLOR):
                    nonlocal y_offset
                    info_text = self.font.render(f"{label}: {value}", True, color)
                    surface.blit(info_text, (self.sim_area_width + 20, y_offset))
                    y_offset += 20
                
                draw_agent_info("ID", selected_agent.id)
                draw_agent_info("父代ID", selected_agent.parent_id if selected_agent.parent_id else "N/A")
                draw_agent_info("基因型ID", selected_agent.genotype_id)
                draw_agent_info("能量", f"{selected_agent.energy:.2f}")
                draw_agent_info("位置", f"({selected_agent.position.x:.1f}, {selected_agent.position.y:.1f})")
                draw_agent_info("隐藏节点数", selected_agent.n_hidden)
                draw_agent_info("连接数", selected_agent.n_connections)
                draw_agent_info("计算深度", selected_agent.computation_depth)
                
                # 如果是变异体，特殊标记
                if selected_agent.is_mutant:
                    draw_agent_info("变异体", "是", HIGHLIGHT_COLOR)
                
                y_offset += 10
        
        # 绘制事件日志
        events_title = self.font.render("--- 最近事件 ---", True, HIGHLIGHT_COLOR)
        surface.blit(events_title, (self.sim_area_width + 20, self.screen_height - 300))
        
        # 获取最近的10个事件
        recent_events = []
        
        # 获取当前帧的事件
        if current_frame in self.event_data:
            recent_events.extend([(current_frame, event) for event in self.event_data[current_frame]])
        
        # 查找当前帧之前的事件
        prev_frame_idx = self.current_frame_idx - 1
        while len(recent_events) < 10 and prev_frame_idx >= 0:
            frame = self.frames[prev_frame_idx]
            if frame in self.event_data:
                for event in self.event_data[frame]:
                    recent_events.append((frame, event))
            prev_frame_idx -= 1
            
        # 保留最近的10个事件
        recent_events = recent_events[:10]
        
        # 绘制事件列表
        event_y = self.screen_height - 270
        for frame, event in recent_events:
            # 根据事件类型选择颜色
            color = EVENT_COLORS.get(event['type'], TEXT_COLOR)
            
            # 简化事件详情显示
            details_text = ""
            if event['type'] == 'MUTATION':
                details_text = f"父:{event['details'].get('parent_id', '?')}->子:{event['details'].get('child_id', '?')}"
            elif event['type'] == 'AGENT_DEATH':
                details_text = f"ID:{event['details'].get('agent_id', '?')}, 原因:{event['details'].get('reason', '未知')}"
            elif event['type'] == 'PREDATION':
                details_text = f"捕食者:{event['details'].get('pred_id', '?')}->被捕食:{event['details'].get('prey_id', '?')}"
            elif 'message' in event['details']:
                details_text = event['details']['message']
            elif 'count' in event['details']:
                details_text = f"数量:{event['details']['count']}"
            
            # 绘制事件信息
            event_text = self.font.render(f"F{frame}: {event['type']} {details_text}", True, color)
            surface.blit(event_text, (self.sim_area_width + 20, event_y))
            event_y += 20

# --- 主函数 ---
def main():
    # 设置窗口位置居中
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    # 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((INITIAL_SCREEN_WIDTH, INITIAL_SCREEN_HEIGHT), 
                                     pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("ECE 回放器")
    
    # 使用硬件加速
    if pygame.display.get_driver() == 'windows':
        # 在Windows上尝试使用DirectX
        os.environ['SDL_VIDEODRIVER'] = 'directx'
    
    # 设置字体 - 使用与原模拟相同的中文字体处理
    try: 
        font = pygame.font.SysFont("simhei", 16)  # 优先使用中文黑体
    except pygame.error: 
        font = pygame.font.SysFont(None, 22)  # 如果没有中文字体，使用默认字体
    
    # 设置时钟
    clock = pygame.time.Clock()
    
    # 获取当前屏幕尺寸
    current_screen_width, current_screen_height = screen.get_size()
    
    # 创建回放控制器
    controller = ReplayController(current_screen_width, current_screen_height, font)
    controller.frame_time = 1 / 10  # 默认每秒10帧
    controller.frame_advance_time = 0
    
    # 选择日志文件
    if not controller.select_log_file():
        print("未选择有效的日志文件，程序退出")
        pygame.quit()
        return
    
    running = True
    
    while running:
        dt = clock.tick(FRAME_RATE) / 1000.0
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # 窗口大小调整
            if event.type == pygame.VIDEORESIZE:
                screen_width, screen_height = event.size
                screen = pygame.display.set_mode((screen_width, screen_height), 
                                               pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
                controller.update_layout(screen_width, screen_height)
                
            # 处理控制器事件
            controller.handle_event(event, pygame.mouse.get_pos())
        
        # 更新回放状态
        controller.update(dt)
        
        # 绘制界面
        controller.draw(screen)
        
        # 更新显示
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main() 