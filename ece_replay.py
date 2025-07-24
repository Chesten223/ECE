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
import base64
import math # Added for hypot

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
TOOLTIP_COLOR = (0, 0, 0, 200)
TAB_ACTIVE_COLOR = (120, 170, 220)
TAB_INACTIVE_COLOR = (70, 70, 90)
BUTTON_HOVER_COLOR = (90, 130, 180)
DROPDOWN_BG_COLOR = (50, 50, 70, 240)
DROPDOWN_HOVER_COLOR = (80, 80, 120)
DROPDOWN_SELECTED_COLOR = (100, 120, 200)
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

# --- 场景可视化系统 ---
class Field:
    def __init__(self, size, color, name):
        self.size = size
        self.color = color
        self.name = name
        self.grid = np.zeros((size, size), dtype=np.float32)
        # 渲染缓存
        self.last_render_surface = None
        self.last_camera_params = None

    def set_data(self, encoded_data):
        """从Base64编码的字符串中加载场数据"""
        try:
            # 解码数据
            binary_data = base64.b64decode(encoded_data)
            # 转换回numpy数组
            grid_data = np.frombuffer(binary_data, dtype=np.float32)
            # 重塑数组形状
            self.grid = grid_data.reshape((self.size, self.size))
            # 清除渲染缓存
            self.last_render_surface = None
        except Exception as e:
            print(f"加载场数据错误: {e}")

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
        # 【修改点】删除这一行
        # self.computation_depth = agent_data['computation_depth']
        try:
            self.gene = json.loads(agent_data['gene_string'].replace("'", '"'))
        except:
            self.gene = {"connections": []}
        self.radius = 2.0
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
        self.field_log_path = None  # 添加场景数据日志路径
        self.signal_types_path = None  # 添加信号类型日志路径
        self.frame_data = {}  # 帧号到智能体数据的映射
        self.event_data = {}  # 帧号到事件的映射
        self.field_data = {}  # 帧号到场景数据的映射
        self.signal_types = []  # 存储信号类型列表
        self.frames = []      # 所有帧号
        self.current_frame_idx = 0
        self.playing = False
        self.playback_speed = 1.0
        self.selected_agent_id = None
        self.genotype_colors = {}
        self.font = font  # 使用传入的字体
        
        # 视图模式: 0=所有场, 1=能量场, 2=危险场, 3+=信号场, 5=无场景
        self.view_mode = 1  # 默认显示能量场
        
        # UI选项卡控制
        self.info_tab = "stats"  # 当前选中的信息面板选项卡: "stats"=统计信息, "neural"=神经网络, "events"=事件日志
        
        # 性能选项
        self.show_all_agents = True  # 是否显示所有智能体
        self.max_visible_agents = 200  # 最大显示的智能体数量
        self.agent_render_distance = float('inf')  # 智能体渲染距离
        
        # 智能体缓存 - 防止闪烁
        self.agents_cache = {}  # 帧号到智能体列表的缓存

        # 创建基础场对象
        self.fields = [
            Field(WORLD_SIZE, 1, "Nutrient/Energy"),  # 营养/能量场（绿色）
            Field(WORLD_SIZE, 0, "Hazard"),          # 危险/障碍场（红色）
        ]
        
        # 信号场将在加载数据时动态添加
        
        # 事件列表滚动
        self.event_scroll_y = 0
        self.event_scroll_max = 0
        self.event_visible_items = 15  # 增加可显示的事件数量
        self.event_item_height = 20
        self.event_list_events = []
        self.event_scroll_dragging = False
        self.max_stored_events = 50  # 最多保存的事件数量
        
        # UI交互状态
        self.hover_element = None  # 当前鼠标悬停的元素
        self.ui_elements = {}  # 界面元素字典，用于追踪工具提示
        
        # 下拉列表状态
        self.dropdown_open = False  # 信号场下拉列表是否打开
        self.dropdown_items = []    # 下拉列表项
        self.dropdown_hover_idx = -1  # 当前悬停的下拉列表项索引
        
        # 计算界面布局
        self.update_layout(screen_width, screen_height)
        
        # 初始化相机
        self.camera = Camera(self.sim_area_width, screen_height)
        self.frame_advance_time = 0  # 确保这个属性被初始化
        
        # 统一的速度控制
        self.playback_rate = 1.0  # 播放速率，结合了帧率、跳帧和速度
    
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
        
        # 事件列表区域
        self.event_list_height = 350  # 增加事件列表区域高度
        self.event_list_rect = pygame.Rect(
            self.sim_area_width + 20, 
            screen_height - self.event_list_height,
            self.info_panel_width - 40, 
            self.event_list_height - 30
        )
        
        # 事件列表滚动条
        scrollbar_width = 10
        self.event_scrollbar_rect = pygame.Rect(
            self.event_list_rect.right - scrollbar_width,
            self.event_list_rect.top,
            scrollbar_width,
            self.event_list_rect.height
        )
        
        # 滚动条滑块
        self.event_scrollbar_button_rect = pygame.Rect(
            self.event_scrollbar_rect.x,
            self.event_scrollbar_rect.y,
            scrollbar_width,
            50  # 初始高度，会根据内容动态调整
        )
        
        # UI元素及其工具提示
        self.ui_elements = {
            "play_button": {
                "rect": self.play_button_rect,
                "tooltip": "播放 (空格键)"
            },
            "pause_button": {
                "rect": self.pause_button_rect,
                "tooltip": "暂停 (空格键)"
            },
            "speed_up_button": {
                "rect": self.speed_up_rect,
                "tooltip": "加快播放速度"
            },
            "speed_down_button": {
                "rect": self.speed_down_rect,
                "tooltip": "降低播放速度"
            },
            "slider": {
                "rect": self.slider_rect,
                "tooltip": "拖动以跳转到特定帧"
            },
        }
        
        # 基础视图模式按钮区域
        button_width = 80
        button_height = 25
        button_margin = 10
        
        # 计算信号场下拉列表的位置（置顶）
        signal_dropdown_y = 75  # 标题下方
        
        # 信号场下拉列表
        dropdown_width = self.info_panel_width - 80
        self.signal_dropdown_rect = pygame.Rect(
            self.sim_area_width + 20, 
            signal_dropdown_y, 
            dropdown_width, 
            button_height
        )
        
        # 添加下拉列表到UI元素
        self.ui_elements["signal_dropdown"] = {
            "rect": self.signal_dropdown_rect,
            "tooltip": "选择要显示的信号场"
        }
        
        # 准备下拉列表项
        self.dropdown_items = []
        if hasattr(self, 'signal_types') and self.signal_types:
            # 基础选项
            self.dropdown_items = [
                {"text": "不显示信号场", "value": -1},
            ]
            
            # 添加每个信号类型
            for i, signal_type in enumerate(self.signal_types):
                self.dropdown_items.append({
                    "text": f"信号场 {i+1}: {signal_type}",
                    "value": i + 3  # 信号场从索引3开始
                })
            
            # 下拉列表展开区域 - 向下展开
            item_height = 25
            dropdown_height = len(self.dropdown_items) * item_height
            self.dropdown_list_rect = pygame.Rect(
                self.signal_dropdown_rect.x,
                self.signal_dropdown_rect.bottom,
                self.signal_dropdown_rect.width,
                dropdown_height
            )
        
        # 计算视图模式按钮的位置（在统计信息之后）
        basic_modes_y = signal_dropdown_y + button_height + 100  # 留出足够空间给统计信息
        
        # 视图模式按钮
        button_x = self.sim_area_width + 20
        mode_names = ["所有场", "能量场", "危险场", "无场景"]
        mode_descriptions = [
            "显示所有场景数据",
            "仅显示能量/营养场 (绿色)",
            "仅显示危险/障碍场 (红色)",
            "不显示任何场景数据"
        ]
        
        for i, (name, desc) in enumerate(zip(mode_names, mode_descriptions)):
            mode_idx = i if i < 3 else 5  # 无场景是索引5
            button_rect = pygame.Rect(button_x, basic_modes_y, button_width, button_height)
            self.ui_elements[f"view_mode_{mode_idx}"] = {
                "rect": button_rect,
                "tooltip": f"{name}: {desc} (按{i+1}键切换)"
            }
            
            button_x += button_width + button_margin
            if button_x + button_width > self.sim_area_width + self.info_panel_width - 20:
                button_x = self.sim_area_width + 20
                basic_modes_y += button_height + 5
        
        # 选项卡区域 (更大的点击区域)
        tab_width = (self.info_panel_width - 40) // 3
        tab_height = 30
        
        # 计算选项卡按钮的Y坐标 - 在视图模式按钮之后
        tab_y = basic_modes_y + button_height + 20
        
        # 定义选项卡 (使用更大的点击区域)
        tab_rect_expansion = 10  # 向各方向扩展的像素数
        
        # 统计信息选项卡
        stats_tab_visible_rect = pygame.Rect(self.sim_area_width + 20, tab_y, tab_width, tab_height)
        stats_tab_clickable_rect = pygame.Rect(
            stats_tab_visible_rect.x - tab_rect_expansion,
            stats_tab_visible_rect.y - tab_rect_expansion,
            stats_tab_visible_rect.width + tab_rect_expansion,
            stats_tab_visible_rect.height + tab_rect_expansion * 2
        )
        
        self.ui_elements["stats_tab"] = {
            "rect": stats_tab_clickable_rect,
            "visible_rect": stats_tab_visible_rect,
            "tooltip": "查看选中智能体的详细信息",
            "tab_id": "stats"
        }
        
        # 神经网络选项卡
        neural_tab_visible_rect = pygame.Rect(self.sim_area_width + 20 + tab_width, tab_y, tab_width, tab_height)
        neural_tab_clickable_rect = pygame.Rect(
            neural_tab_visible_rect.x - tab_rect_expansion,
            neural_tab_visible_rect.y - tab_rect_expansion,
            neural_tab_visible_rect.width + tab_rect_expansion * 2,
            neural_tab_visible_rect.height + tab_rect_expansion * 2
        )
        
        self.ui_elements["neural_tab"] = {
            "rect": neural_tab_clickable_rect,
            "visible_rect": neural_tab_visible_rect,
            "tooltip": "查看选中智能体的神经网络结构",
            "tab_id": "neural"
        }
        
        # 事件日志选项卡
        events_tab_visible_rect = pygame.Rect(self.sim_area_width + 20 + tab_width * 2, tab_y, tab_width, tab_height)
        events_tab_clickable_rect = pygame.Rect(
            events_tab_visible_rect.x - tab_rect_expansion,
            events_tab_visible_rect.y - tab_rect_expansion,
            events_tab_visible_rect.width + tab_rect_expansion,
            events_tab_visible_rect.height + tab_rect_expansion * 2
        )
        
        self.ui_elements["events_tab"] = {
            "rect": events_tab_clickable_rect,
            "visible_rect": events_tab_visible_rect,
            "tooltip": "查看系统事件日志",
            "tab_id": "events"
        }
        
        # 如果相机已初始化，更新相机渲染尺寸
        if hasattr(self, 'camera'):
            self.camera.update_render_size(self.sim_area_width, screen_height)
    
    def handle_event(self, event, mouse_pos):
        """处理用户输入事件"""
        # 处理相机事件
        self.camera.handle_event(event, mouse_pos)
        
        # 处理UI控件
        if event.type == pygame.MOUSEMOTION:
            # 检测鼠标悬停在哪个UI元素上
            self.hover_element = None
            for element_id, element_data in self.ui_elements.items():
                if is_point_in_rect(mouse_pos, element_data["rect"]):
                    self.hover_element = element_id
                    break
            
            # 处理下拉列表项悬停
            if self.dropdown_open and hasattr(self, 'dropdown_list_rect'):
                if is_point_in_rect(mouse_pos, self.dropdown_list_rect):
                    # 计算悬停的项目索引
                    rel_y = mouse_pos[1] - self.dropdown_list_rect.y
                    item_height = self.dropdown_list_rect.height / len(self.dropdown_items)
                    self.dropdown_hover_idx = int(rel_y / item_height)
                else:
                    self.dropdown_hover_idx = -1
            
            # 处理进度条拖动
            if self.slider_dragging:
                # 计算拖动位置对应的帧索引
                progress = (mouse_pos[0] - self.slider_rect.x) / self.slider_rect.width
                progress = max(0, min(1, progress))
                frame_idx = int(progress * (len(self.frames) - 1)) if self.frames else 0
                self.set_frame_index(frame_idx)
            
            # 处理事件滚动条拖动
            if self.event_scroll_dragging:
                # 计算拖动后的滚动位置
                drag_ratio = (mouse_pos[1] - self.event_scrollbar_rect.y) / self.event_scrollbar_rect.height
                drag_ratio = max(0, min(1, drag_ratio))
                self.event_scroll_y = min(self.event_scroll_max, drag_ratio * self.event_scroll_max)
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键点击
                # 检查是否点击信号场下拉列表
                if hasattr(self, 'signal_dropdown_rect') and self.signal_dropdown_rect.collidepoint(mouse_pos):
                    self.dropdown_open = not self.dropdown_open
                    return True
                
                # 检查是否点击下拉列表项
                if self.dropdown_open and hasattr(self, 'dropdown_list_rect'):
                    if is_point_in_rect(mouse_pos, self.dropdown_list_rect):
                        # 计算点击的项目索引
                        rel_y = mouse_pos[1] - self.dropdown_list_rect.y
                        item_height = self.dropdown_list_rect.height / len(self.dropdown_items)
                        clicked_idx = int(rel_y / item_height)
                        
                        if 0 <= clicked_idx < len(self.dropdown_items):
                            # 设置选中的视图模式
                            selected_value = self.dropdown_items[clicked_idx]["value"]
                            if selected_value >= 0:  # -1表示不显示信号场
                                self.view_mode = selected_value
                            else:
                                self.view_mode = 5  # 无场景
                            
                            # 关闭下拉列表
                            self.dropdown_open = False
                            return True
                else:
                    # 如果点击了其他地方，关闭下拉列表
                    self.dropdown_open = False
                
                # 检查是否点击进度条
                if self.ui_elements["slider"]["rect"].collidepoint(mouse_pos):
                    # 计算点击位置对应的帧索引
                    progress = (mouse_pos[0] - self.slider_rect.x) / self.slider_rect.width
                    progress = max(0, min(1, progress))
                    frame_idx = int(progress * (len(self.frames) - 1)) if self.frames else 0
                    self.set_frame_index(frame_idx)
                    self.slider_dragging = True
                
                # 检查是否点击播放控制按钮
                if self.ui_elements["play_button"]["rect"].collidepoint(mouse_pos):
                    self.playing = True
                elif self.ui_elements["pause_button"]["rect"].collidepoint(mouse_pos):
                    self.playing = False
                elif self.ui_elements["speed_up_button"]["rect"].collidepoint(mouse_pos):
                    # 加快播放速度
                    self.playback_rate = min(16.0, self.playback_rate * 2.0)  # 最大16倍速
                elif self.ui_elements["speed_down_button"]["rect"].collidepoint(mouse_pos):
                    # 降低播放速度
                    self.playback_rate = max(1.0, self.playback_rate / 2.0)  # 最小1.0倍速
                
                # 处理基础视图模式按钮点击
                for i in range(4):  # 4种基础视图模式
                    mode_idx = i if i < 3 else 5  # 无场景是索引5
                    element_id = f"view_mode_{mode_idx}"
                    if element_id in self.ui_elements and self.ui_elements[element_id]["rect"].collidepoint(mouse_pos):
                        self.view_mode = mode_idx
                        break
                
                # 处理选项卡点击 - 使用扩展的点击区域
                for tab_id in ["stats_tab", "neural_tab", "events_tab"]:
                    if self.ui_elements[tab_id]["rect"].collidepoint(mouse_pos):
                        self.info_tab = self.ui_elements[tab_id]["tab_id"]
                        break
                
                # 检查是否点击模拟区域选择智能体
                if mouse_pos[0] < self.sim_area_width:
                    world_pos = self.camera.screen_to_world(mouse_pos)
                    selected_agent = self.find_agent_at_position(world_pos)
                    if selected_agent:
                        # 如果点击了智能体，设为选中状态
                        for agent in self.get_current_agents():
                            agent.is_selected = (agent.id == selected_agent.id)
                        self.selected_agent_id = selected_agent.id
                    else:
                        # 如果点击空白处，清除选中状态
                        for agent in self.get_current_agents():
                            agent.is_selected = False
                        self.selected_agent_id = None
                
                # 检查是否点击事件滚动条
                if self.event_scrollbar_rect.collidepoint(mouse_pos):
                    # 直接滚动到点击位置
                    click_pos_y = mouse_pos[1] - self.event_scrollbar_rect.y
                    scroll_ratio = click_pos_y / self.event_scrollbar_rect.height
                    self.event_scroll_y = min(self.event_scroll_max, scroll_ratio * self.event_scroll_max)
                
                # 检查是否点击事件滚动条滑块
                if self.event_scrollbar_button_rect.collidepoint(mouse_pos):
                    self.event_scroll_dragging = True
            
            # 滚轮事件 - 用于事件列表滚动
            elif event.button == 4 and self.event_list_rect.collidepoint(mouse_pos):  # 上滚
                self.event_scroll_y = max(0, self.event_scroll_y - self.event_item_height)
            elif event.button == 5 and self.event_list_rect.collidepoint(mouse_pos):  # 下滚
                self.event_scroll_y = min(self.event_scroll_max, self.event_scroll_y + self.event_item_height)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.slider_dragging = False
                self.event_scroll_dragging = False
                
        # 处理键盘事件
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # 空格键切换播放/暂停
                self.playing = not self.playing
            elif event.key == pygame.K_RIGHT:
                # 右箭头前进一帧
                self.next_frame()
            elif event.key == pygame.K_LEFT:
                # 左箭头后退一帧
                self.prev_frame()
            elif event.key == pygame.K_UP:
                # 上箭头加快播放速度
                self.playback_rate = min(16.0, self.playback_rate * 1.5)
            elif event.key == pygame.K_DOWN:
                # 下箭头降低播放速度
                self.playback_rate = max(1.0, self.playback_rate / 1.5)
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9):
                # 数字键1-9切换视图模式
                key_num = event.key - pygame.K_1
                if key_num == 0:  # 1键 - 所有场
                    self.view_mode = 0
                elif key_num == 1:  # 2键 - 能量场
                    self.view_mode = 1
                elif key_num == 2:  # 3键 - 危险场
                    self.view_mode = 2
                elif key_num == 3:  # 4键 - 无场景
                    self.view_mode = 5
                elif 4 <= key_num < 4 + len(self.signal_types):  # 5-9键 - 信号场
                    self.view_mode = key_num - 1
            elif event.key == pygame.K_PAGEUP:
                # PageUp向上滚动多行
                scroll_amount = min(5 * self.event_item_height, self.event_scroll_y)
                self.event_scroll_y -= scroll_amount
            elif event.key == pygame.K_PAGEDOWN:
                # PageDown向下滚动多行
                scroll_amount = min(5 * self.event_item_height, self.event_scroll_max - self.event_scroll_y)
                self.event_scroll_y += scroll_amount
            elif event.key == pygame.K_HOME:
                # Home键滚动到顶部
                self.event_scroll_y = 0
            elif event.key == pygame.K_END:
                # End键滚动到底部
                self.event_scroll_y = self.event_scroll_max
            elif event.key == pygame.K_a:
                # A键切换是否显示所有智能体
                self.show_all_agents = not self.show_all_agents
                # 清除缓存以应用新的显示设置
                self.agents_cache = {}
            elif event.key == pygame.K_ESCAPE:
                # ESC键关闭下拉列表
                if self.dropdown_open:
                    self.dropdown_open = False
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
    
    def get_current_frame(self):
        """获取当前帧号"""
        if not self.frames:
            return 0
        return self.frames[self.current_frame_idx]
    
    def get_current_agents(self):
        """获取当前帧的智能体数据（带性能优化和缓存）"""
        current_frame = self.get_current_frame()
        
        # 如果缓存中有当前帧的数据，直接返回
        if current_frame in self.agents_cache:
            return self.agents_cache[current_frame]
            
        if current_frame in self.frame_data:
            agents = []
            agent_data_list = self.frame_data[current_frame]
            
            # 如果不显示所有智能体且数量超过限制，则随机采样
            if not self.show_all_agents and len(agent_data_list) > self.max_visible_agents:
                # 确保选中的智能体总是可见
                selected_agent_data = None
                filtered_agents = []
                
                for agent_data in agent_data_list:
                    if agent_data['agent_id'] == self.selected_agent_id:
                        selected_agent_data = agent_data
                    else:
                        filtered_agents.append(agent_data)
                
                # 随机选择一部分智能体
                import random
                sampled_agents = random.sample(filtered_agents, 
                                              min(self.max_visible_agents - 1, len(filtered_agents)))
                
                # 如果有选中的智能体，确保它被添加
                if selected_agent_data:
                    sampled_agents.append(selected_agent_data)
                
                agent_data_list = sampled_agents
            
            # 创建所有可见智能体
            for agent_data in agent_data_list:
                agent = ReplayAgent(agent_data)
                if agent.id == self.selected_agent_id:
                    agent.is_selected = True
                agents.append(agent)
            
            # 将结果存入缓存
            self.agents_cache[current_frame] = agents
            
            # 如果缓存太大，删除最老的条目
            if len(self.agents_cache) > 20:  # 保留20帧的缓存
                oldest_frame = min(self.agents_cache.keys())
                del self.agents_cache[oldest_frame]
                
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
        if not self.frames:
            return
        self.current_frame_idx = max(0, min(index, len(self.frames) - 1))
        # 更新选中智能体状态
        current_agents = self.get_current_agents()
        for agent in current_agents:
            agent.is_selected = (agent.id == self.selected_agent_id)
            
        # 更新事件列表
        self.update_event_list()
        
        # 更新场景数据
        current_frame = self.get_current_frame()
        self.update_field_data_for_frame(current_frame)
    
    def update_event_list(self):
        """更新事件列表数据"""
        # 清空之前的事件列表
        self.event_list_events = []
        
        # 获取当前帧的事件
        current_frame = self.get_current_frame()
        if current_frame in self.event_data:
            self.event_list_events.extend([(current_frame, event) for event in self.event_data[current_frame]])
        
        # 查找当前帧之前的事件
        prev_frame_idx = self.current_frame_idx - 1
        while len(self.event_list_events) < self.max_stored_events and prev_frame_idx >= 0:
            frame = self.frames[prev_frame_idx]
            if frame in self.event_data:
                for event in self.event_data[frame]:
                    self.event_list_events.append((frame, event))
                    if len(self.event_list_events) >= self.max_stored_events:
                        break
            prev_frame_idx -= 1
        
        # 计算滚动条最大值
        total_items = len(self.event_list_events)
        visible_items = self.event_visible_items
        self.event_scroll_max = max(0, (total_items - visible_items) * self.event_item_height)
        
        # 调整滚动条滑块大小和位置
        if total_items > 0:
            # 滑块高度与可见部分占总内容的比例相同
            scrollbar_height = max(20, self.event_scrollbar_rect.height * visible_items / total_items)
            self.event_scrollbar_button_rect.height = int(scrollbar_height)
            
            # 滑块位置
            if self.event_scroll_max > 0:
                scrollbar_pos = (self.event_scroll_y / self.event_scroll_max) * (self.event_scrollbar_rect.height - scrollbar_height)
            else:
                scrollbar_pos = 0
                
            self.event_scrollbar_button_rect.y = int(self.event_scrollbar_rect.y + scrollbar_pos)
    
    def update(self, dt):
        """更新回放状态"""
        # 如果正在播放且有帧数据
        if self.playing and self.frames:
            # 根据播放速度累积时间
            self.frame_advance_time += dt * self.playback_rate
            
            # 每秒基准帧率为10
            base_frame_time = 1 / 10
            
            # 如果累积的时间足够前进一帧
            while self.frame_advance_time >= base_frame_time:
                self.frame_advance_time -= base_frame_time
                
                # 前进一帧
                if not self.next_frame():
                    self.playing = False  # 播放到末尾时停止
                    break
        
        # 更新事件列表
        self.update_event_list()
    
    def get_current_field_data(self):
        """获取当前帧的场景数据"""
        frame = self.get_current_frame()
        return self.field_data.get(frame, {})
    
    def update_field_data_for_frame(self, frame):
        """为特定帧更新场景数据"""
        # 找到离当前帧最近的包含场景数据的帧
        nearest_frame = None
        for f in sorted(self.field_data.keys(), reverse=True):
            if f <= frame:
                nearest_frame = f
                break
        
        if nearest_frame is not None:
            frame_fields = self.field_data[nearest_frame]
            for field_idx, encoded_data in frame_fields.items():
                if field_idx < len(self.fields):
                    self.fields[field_idx].set_data(encoded_data)
    
    def select_log_file(self):
        """选择日志文件目录"""
        root = tk.Tk()
        root.withdraw()
        log_dir = filedialog.askdirectory(title="选择日志目录")
        if log_dir:
            self.sim_log_path = os.path.join(log_dir, "simulation_log.csv")
            self.event_log_path = os.path.join(log_dir, "event_log.csv")
            self.field_log_path = os.path.join(log_dir, "field_log.csv")  # 添加场景数据日志路径
            self.signal_types_path = os.path.join(log_dir, "signal_types.json") # 添加信号类型日志路径
            
            if os.path.exists(self.sim_log_path) and os.path.exists(self.event_log_path):
                self.load_data()
                return True
        
        return False
    
    def load_data(self):
        """加载日志数据"""
        print("正在加载模拟数据...")
        start_time = time.time()
        
        # 使用更简单但高效的方法加载数据
        print("加载智能体模拟数据...")
        # 允许NA值并转换为字符串
        sim_df = pd.read_csv(self.sim_log_path)
        
        print("加载事件数据...")
        event_df = pd.read_csv(self.event_log_path)

        # 尝试加载场景数据
        self.field_data = {}
        try:
            print("加载场景数据...")
            # 直接加载并过滤场景数据，只保留关键帧
            field_df = pd.read_csv(self.field_log_path)
            # 更激进的数据减少：每20帧保留一帧，大幅减少内存占用
            frame_mod = 20  
            field_df = field_df[field_df['frame'] % frame_mod == 0]

            # 进一步优化：仅保留必要字段
            field_df = field_df[['frame', 'field_type', 'data']]
            
            has_field_data = True
            print(f"场景数据加载完成，保留了{len(field_df)}行数据...")
        except Exception as e:
            print(f"未找到场景数据或加载出错: {e}")
            has_field_data = False
        
        # 尝试加载信号类型数据
        self.signal_types = []
        try:
            if os.path.exists(self.signal_types_path):
                with open(self.signal_types_path, 'r') as f:
                    self.signal_types = json.load(f)
                print(f"找到信号类型数据，共有{len(self.signal_types)}种信号类型")
                
                # 动态创建信号场
                colors = [2, 0, 1, 0, 2, 1]  # 蓝、红、绿、红、蓝、绿
                for i, signal_type in enumerate(self.signal_types):
                    color_idx = i % len(colors)
                    field = Field(WORLD_SIZE, colors[color_idx], signal_type)
                    self.fields.append(field)
                    print(f"创建信号场: {signal_type}, 颜色索引: {colors[color_idx]}")
        except Exception as e:
            print(f"未找到信号类型数据或加载出错: {e}")
        
        # 转换为字典以便快速查询
        self.frame_data = {}
        self.event_data = {}
        
        # 使用更高效的分组处理模拟数据
        print("处理智能体数据...")
        # 使用dataframe的groupby功能，更高效
        for frame, group in sim_df.groupby('frame'):
            frame_int = int(frame)
            self.frame_data[frame_int] = []
            
            # 使用迭代器处理每行，减少内存使用
            for _, row in group.iterrows():
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
                    # 【修改点】删除这一行
                    # 'computation_depth': row['computation_depth'],
                    'gene_string': row['gene_string']
                }
                self.frame_data[frame_int].append(agent_data)
        
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
        
        # 处理场景数据
        if has_field_data:
            # 创建字段名称到场索引的映射
            field_name_to_idx = {
                "Nutrient/Energy": 0,
                "Hazard": 1,
            }
            
            # 添加信号场到映射
            for i, signal_type in enumerate(self.signal_types):
                field_name_to_idx[signal_type] = i + 2
            
            # 使用更高效的方式处理场景数据
            field_groups = field_df.groupby('frame')
            for frame, group in field_groups:
                frame_int = int(frame)
                frame_fields = {}
                
                for _, row in group.iterrows():
                    field_type = row['field_type']
                    encoded_data = row['data']
                    
                    field_idx = field_name_to_idx.get(field_type)
                    if field_idx is not None and field_idx < len(self.fields):
                        frame_fields[field_idx] = encoded_data
                
                if frame_fields:
                    self.field_data[frame_int] = frame_fields
        
        self.frames = sorted(list(self.frame_data.keys()))
        self.current_frame_idx = 0
        
        if self.frames and self.field_data:
            self.update_field_data_for_frame(self.get_current_frame())
        
        print(f"数据加载完成，共有{len(self.frames)}帧，耗时{time.time() - start_time:.2f}秒")
        
        self.update_event_list()
        
    def update_signal_types(self):
        """更新信号类型列表 - 这个方法是为了兼容性添加的"""
        pass

    def draw(self, screen):
        """绘制回放界面"""
        # 绘制背景
        screen.fill(BACKGROUND_COLOR)
        
        # 创建模拟区域的表面
        sim_surface = pygame.Surface((self.sim_area_width, self.screen_height))
        sim_surface.fill(BACKGROUND_COLOR)
        
        # 根据视图模式绘制场
        current_frame = self.get_current_frame()
        if current_frame in self.field_data:
            if self.view_mode == 0:
                # 显示所有场
                for field in self.fields:
                    field.draw(sim_surface, self.camera)
            elif 1 <= self.view_mode <= 2:
                # 显示基础场（能量场或危险场）
                self.fields[self.view_mode - 1].draw(sim_surface, self.camera, alpha=200)
            elif 3 <= self.view_mode < 3 + len(self.signal_types):
                # 显示信号场
                field_idx = self.view_mode - 1  # 因为信号场从索引2开始
                if field_idx < len(self.fields):
                    self.fields[field_idx].draw(sim_surface, self.camera, alpha=200)
            # 视图模式5是无场景，不需要绘制
        
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
        
        # 绘制控制按钮和标签
        control_label = self.font.render("播放控制:", True, TEXT_COLOR)
        screen.blit(control_label, (10, self.play_button_rect.y - 20))
        
        # 绘制播放控制按钮
        button_colors = {
            "play_button": (100, 200, 100) if self.playing else (150, 150, 150),
            "pause_button": (200, 100, 100) if not self.playing else (150, 150, 150),
            "speed_up_button": (150, 150, 150),
            "speed_down_button": (150, 150, 150)
        }
        
        # 鼠标悬停高亮
        for button_id, rect_key in [
            ("play_button", "play_button_rect"),
            ("pause_button", "pause_button_rect"),
            ("speed_up_button", "speed_up_rect"),
            ("speed_down_button", "speed_down_rect")
        ]:
            rect = getattr(self, rect_key)
            base_color = button_colors[button_id]
            
            # 绘制按钮背景
            if self.hover_element == button_id:
                # 悬停时混合一些高亮色
                hover_color = (
                    min(255, base_color[0] + 40),
                    min(255, base_color[1] + 40),
                    min(255, base_color[2] + 40)
                )
                pygame.draw.rect(screen, hover_color, rect)
            else:
                pygame.draw.rect(screen, base_color, rect)
            
            # 绘制按钮边框
            pygame.draw.rect(screen, (100, 100, 100), rect, 1)
        
        # 绘制按钮文字和图标
        small_font = pygame.font.SysFont(None, 20)
        play_text = small_font.render("▶", True, (0, 0, 0))
        pause_text = small_font.render("❚❚", True, (0, 0, 0))
        speed_up_text = small_font.render(">>", True, (0, 0, 0))
        speed_down_text = small_font.render("<<", True, (0, 0, 0))
        
        screen.blit(play_text, (self.play_button_rect.x + 8, self.play_button_rect.y + 3))
        screen.blit(pause_text, (self.pause_button_rect.x + 8, self.pause_button_rect.y + 3))
        screen.blit(speed_up_text, (self.speed_up_rect.x + 5, self.speed_up_rect.y + 3))
        screen.blit(speed_down_text, (self.speed_down_rect.x + 5, self.speed_down_rect.y + 3))
        
        # 绘制按钮标签
        button_labels = [("播放", self.play_button_rect), ("暂停", self.pause_button_rect), 
                        ("加速", self.speed_up_rect), ("减速", self.speed_down_rect)]
        
        for label, rect in button_labels:
            label_text = self.font.render(label, True, TEXT_COLOR)
            label_x = rect.centerx - label_text.get_width() // 2
            label_y = rect.bottom + 5
            screen.blit(label_text, (label_x, label_y))
        
        # 绘制当前帧信息和播放速度
        current_frame = self.get_current_frame()
        total_frames = self.frames[-1] if self.frames else 0
        
        stats_x = self.slider_rect.x + self.slider_rect.width + 20
        
        frame_text = self.font.render(f"帧: {current_frame} / {total_frames}", True, TEXT_COLOR)
        screen.blit(frame_text, (stats_x, self.slider_rect.y))
        
        # 绘制播放速度文本
        speed_text = self.font.render(f"播放速度: {self.playback_rate:.2f}x (上下箭头调整)", True, HIGHLIGHT_COLOR)
        screen.blit(speed_text, (stats_x, self.slider_rect.y - 25))
        
        # 绘制性能信息
        agent_count = len(self.get_current_agents())
        if self.show_all_agents:
            agent_text = self.font.render(f"显示全部智能体 ({agent_count}个) (A键切换)", True, TEXT_COLOR)
        else:
            agent_text = self.font.render(f"限制显示 (上限{self.max_visible_agents}个) (A键切换)", True, TEXT_COLOR)
        screen.blit(agent_text, (stats_x, self.slider_rect.y - 50))
        
        # 绘制操作提示
        tip_text = self.font.render("左键: 选择智能体 | 右键: 平移 | 滚轮: 缩放", True, HIGHLIGHT_COLOR)
        screen.blit(tip_text, (10, self.slider_rect.y - 60))
    
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
        
        y_offset = 50  # 从标题下方开始
        
        # 绘制信号场下拉列表（移到顶部）
        if hasattr(self, 'signal_types') and self.signal_types:
            signal_title = self.font.render("信号场选择:", True, TEXT_COLOR)
            surface.blit(signal_title, (self.sim_area_width + 20, y_offset))
            y_offset += 25
            
            # 绘制下拉列表按钮
            dropdown_rect = self.signal_dropdown_rect
            
            # 根据是否悬停绘制不同颜色
            if self.hover_element == "signal_dropdown":
                pygame.draw.rect(surface, BUTTON_HOVER_COLOR, dropdown_rect)
            else:
                pygame.draw.rect(surface, TAB_INACTIVE_COLOR, dropdown_rect)
                
            # 绘制边框
            pygame.draw.rect(surface, (150, 150, 150), dropdown_rect, 1)
            
            # 查找当前选中的信号场
            selected_text = "选择信号场..."
            for item in self.dropdown_items:
                if item["value"] == self.view_mode:
                    selected_text = item["text"]
                    break
                    
            # 如果没有匹配的选项，且当前是无场景模式
            if self.view_mode == 5 and selected_text == "选择信号场...":
                selected_text = "不显示信号场"
                
            # 绘制当前选中的文本
            text_surface = self.font.render(selected_text, True, TEXT_COLOR)
            text_rect = text_surface.get_rect(midleft=(dropdown_rect.x + 10, dropdown_rect.centery))
            surface.blit(text_surface, text_rect)
            
            # 绘制下拉箭头
            arrow_points = [
                (dropdown_rect.right - 20, dropdown_rect.centery - 4),
                (dropdown_rect.right - 10, dropdown_rect.centery - 4),
                (dropdown_rect.right - 15, dropdown_rect.centery + 4)
            ]
            pygame.draw.polygon(surface, TEXT_COLOR, arrow_points)
            
            # 如果下拉列表打开，绘制列表项
            if self.dropdown_open:
                # 绘制下拉列表背景
                if hasattr(self, 'dropdown_list_rect'):
                    pygame.draw.rect(surface, DROPDOWN_BG_COLOR, self.dropdown_list_rect)
                    pygame.draw.rect(surface, (150, 150, 150), self.dropdown_list_rect, 1)
                    
                    # 绘制列表项
                    item_height = self.dropdown_list_rect.height / len(self.dropdown_items)
                    for i, item in enumerate(self.dropdown_items):
                        item_rect = pygame.Rect(
                            self.dropdown_list_rect.x,
                            self.dropdown_list_rect.y + i * item_height,
                            self.dropdown_list_rect.width,
                            item_height
                        )
                        
                        # 根据悬停和选中状态绘制不同颜色
                        if i == self.dropdown_hover_idx:
                            pygame.draw.rect(surface, DROPDOWN_HOVER_COLOR, item_rect)
                        elif item["value"] == self.view_mode:
                            pygame.draw.rect(surface, DROPDOWN_SELECTED_COLOR, item_rect)
                        
                        # 绘制项目文本
                        item_text = self.font.render(item["text"], True, TEXT_COLOR)
                        text_rect = item_text.get_rect(midleft=(item_rect.x + 10, item_rect.centery))
                        surface.blit(item_text, text_rect)
            
            y_offset += dropdown_rect.height + 20
        
        # 绘制当前帧统计信息
        agents = self.get_current_agents()
        current_frame = self.get_current_frame()
        
        # 智能体统计
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
        
        # 绘制视图模式切换标题（移到后面）
        view_title = self.font.render("基础视图模式:", True, TEXT_COLOR)
        surface.blit(view_title, (self.sim_area_width + 20, y_offset))
        y_offset += 25
        
        # 绘制基础视图模式按钮
        basic_modes = ["所有场", "能量场", "危险场", "无场景"]
        button_width = 80
        button_height = 25
        button_margin = 10
        button_x = self.sim_area_width + 20
        
        for i, mode_name in enumerate(basic_modes):
            mode_idx = i if i < 3 else 5  # 无场景是索引5
            mode_idx = i if i < 3 else 5  # 无场景是索引5
            element_id = f"view_mode_{mode_idx}"
            button_rect = self.ui_elements[element_id]["rect"]
            
            # 当前选中的模式高亮显示
            if mode_idx == self.view_mode:
                pygame.draw.rect(surface, TAB_ACTIVE_COLOR, button_rect)
                text_color = (0, 0, 0)
            # 鼠标悬停时显示高亮
            elif self.hover_element == element_id:
                pygame.draw.rect(surface, BUTTON_HOVER_COLOR, button_rect)
                text_color = TEXT_COLOR
            else:
                pygame.draw.rect(surface, TAB_INACTIVE_COLOR, button_rect)
                text_color = TEXT_COLOR
                
            pygame.draw.rect(surface, (150, 150, 150), button_rect, 1)  # 边框
            
            # 绘制按钮文字
            mode_text = self.font.render(mode_name, True, text_color)
            text_rect = mode_text.get_rect(center=(button_rect.centerx, button_rect.centery))
            surface.blit(mode_text, text_rect)
            
            # 标注快捷键
            key_text = self.font.render(f"{i+1}", True, HIGHLIGHT_COLOR)
            surface.blit(key_text, (button_rect.x + 5, button_rect.y + 2))
            
            # 更新下一个按钮的位置
            button_x += button_width + button_margin
            if button_x + button_width > self.sim_area_width + self.info_panel_width - 20:
                button_x = self.sim_area_width + 20
                y_offset += button_height + 5
        
        y_offset += button_height + 20
        
        # 绘制选项卡按钮
        tab_names = ["智能体信息", "神经网络", "事件日志"]
        tab_ids = ["stats_tab", "neural_tab", "events_tab"]
        tab_keys = ["stats", "neural", "events"]
        
        for i, (tab_name, tab_id, tab_key) in enumerate(zip(tab_names, tab_ids, tab_keys)):
            # 获取可见区域和点击区域
            visible_rect = self.ui_elements[tab_id]["visible_rect"]
            clickable_rect = self.ui_elements[tab_id]["rect"]
            
            # 当前选中的选项卡高亮显示
            if self.info_tab == tab_key:
                pygame.draw.rect(surface, TAB_ACTIVE_COLOR, visible_rect)
                text_color = (0, 0, 0)
            # 鼠标悬停时显示高亮
            elif self.hover_element == tab_id:
                pygame.draw.rect(surface, BUTTON_HOVER_COLOR, visible_rect)
                text_color = TEXT_COLOR
            else:
                pygame.draw.rect(surface, TAB_INACTIVE_COLOR, visible_rect)
                text_color = TEXT_COLOR
                
            # 绘制边框 - 选中选项卡的底部边框不绘制
            if self.info_tab == tab_key:
                # 顶部和侧边边框
                pygame.draw.line(surface, (150, 150, 150), visible_rect.topleft, visible_rect.topright)
                pygame.draw.line(surface, (150, 150, 150), visible_rect.topleft, visible_rect.bottomleft)
                pygame.draw.line(surface, (150, 150, 150), visible_rect.topright, visible_rect.bottomright)
            else:
                pygame.draw.rect(surface, (150, 150, 150), visible_rect, 1)
            
            # 绘制选项卡标题
            tab_text = self.font.render(tab_name, True, text_color)
            text_rect = tab_text.get_rect(center=(visible_rect.centerx, visible_rect.centery))
            surface.blit(tab_text, text_rect)
            
            # 显示可点击区域的视觉指示（调试用，正式版可注释掉）
            # pygame.draw.rect(surface, (255, 0, 0, 100), clickable_rect, 1)
        
        # 选项卡内容区域起点
        tab_y = self.ui_elements["stats_tab"]["visible_rect"].bottom
        content_y = tab_y + 10
        content_height = self.screen_height - content_y - 10
        content_rect = pygame.Rect(self.sim_area_width + 20, content_y, self.info_panel_width - 40, content_height)
        
        # 根据当前选项卡绘制不同内容
        if self.info_tab == "stats":
            # 绘制选中智能体信息
            if self.selected_agent_id is not None:
                selected_agent = None
                for agent in agents:
                    if agent.id == self.selected_agent_id:
                        selected_agent = agent
                        break
                
                if selected_agent:
                    agent_title = self.font.render("--- 选中智能体信息 ---", True, HIGHLIGHT_COLOR)
                    surface.blit(agent_title, (self.sim_area_width + 20, content_y))
                    y_offset = content_y + 25
                    
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
                    
                    # 如果是变异体，特殊标记
                    if selected_agent.is_mutant:
                        draw_agent_info("变异体", "是", HIGHLIGHT_COLOR)
                else:
                    no_agent_text = self.font.render("未选择智能体", True, TEXT_COLOR)
                    surface.blit(no_agent_text, (content_rect.centerx - no_agent_text.get_width()//2, content_rect.centery))
            else:
                no_agent_text = self.font.render("点击一个生命体来查看详细信息", True, TEXT_COLOR)
                surface.blit(no_agent_text, (content_rect.centerx - no_agent_text.get_width()//2, content_rect.centery))
        
        elif self.info_tab == "neural":
            # 绘制神经网络
            if self.selected_agent_id is not None:
                selected_agent = None
                for agent in agents:
                    if agent.id == self.selected_agent_id:
                        selected_agent = agent
                        break
                
                if selected_agent:
                    # 绘制神经网络可视化
                    mouse_pos = pygame.mouse.get_pos()
                    draw_neural_network(surface, self.font, selected_agent, content_rect.x, content_rect.y, 
                                       content_rect.width, content_rect.height, mouse_pos)
                else:
                    no_agent_text = self.font.render("未选择智能体", True, TEXT_COLOR)
                    surface.blit(no_agent_text, (content_rect.centerx - no_agent_text.get_width()//2, content_rect.centery))
            else:
                no_agent_text = self.font.render("点击一个生命体来查看神经网络", True, TEXT_COLOR)
                surface.blit(no_agent_text, (content_rect.centerx - no_agent_text.get_width()//2, content_rect.centery))
        
        elif self.info_tab == "events":
            # 绘制事件列表标题
            events_title = self.font.render("--- 最近事件 ---", True, HIGHLIGHT_COLOR)
            surface.blit(events_title, (content_rect.x, content_rect.y))
            
            # 事件列表背景
            event_list_rect = pygame.Rect(content_rect.x, content_rect.y + 30, 
                                         content_rect.width, content_rect.height - 30)
            event_list_bg = pygame.Surface((event_list_rect.width, event_list_rect.height), pygame.SRCALPHA)
            event_list_bg.fill((30, 30, 40, 200))
            surface.blit(event_list_bg, event_list_rect.topleft)
            
            # 更新事件列表矩形
            self.event_list_rect = event_list_rect
            
            # 绘制滚动条
            scrollbar_width = 10
            self.event_scrollbar_rect = pygame.Rect(
                event_list_rect.right - scrollbar_width,
                event_list_rect.top,
                scrollbar_width,
                event_list_rect.height
            )
            pygame.draw.rect(surface, (60, 60, 70), self.event_scrollbar_rect)
            
            # 滚动条滑块
            if self.event_list_events:
                visible_items = self.event_visible_items
                total_items = len(self.event_list_events)
                
                scrollbar_height = max(20, self.event_scrollbar_rect.height * visible_items / total_items)
                self.event_scrollbar_button_rect.height = int(scrollbar_height)
                
                if self.event_scroll_max > 0:
                    scrollbar_pos = (self.event_scroll_y / self.event_scroll_max) * (self.event_scrollbar_rect.height - scrollbar_height)
                else:
                    scrollbar_pos = 0
                    
                self.event_scrollbar_button_rect.y = int(self.event_scrollbar_rect.y + scrollbar_pos)
                self.event_scrollbar_button_rect.x = self.event_scrollbar_rect.x
                self.event_scrollbar_button_rect.width = scrollbar_width
                
                pygame.draw.rect(surface, (120, 120, 140), self.event_scrollbar_button_rect)
            
            # 创建裁剪区域
            clip_rect = surface.get_clip()
            surface.set_clip(event_list_rect)
            
            # 绘制事件
            y_offset = event_list_rect.top - self.event_scroll_y
            for i, (frame, event) in enumerate(self.event_list_events):
                event_y = y_offset + i * self.event_item_height
                
                # 只绘制可见区域内的事件
                if (event_y >= event_list_rect.top - self.event_item_height and 
                    event_y <= event_list_rect.bottom):
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
                    surface.blit(event_text, (event_list_rect.left + 5, event_y))
                    
                    # 当前帧的事件高亮显示
                    if frame == current_frame:
                        pygame.draw.rect(surface, (60, 100, 120, 100), 
                                        pygame.Rect(event_list_rect.left, event_y, 
                                                   event_list_rect.width - self.event_scrollbar_rect.width, 
                                                   self.event_item_height), 1)
            
            # 恢复裁剪区域
            surface.set_clip(clip_rect)
            
            # 绘制事件列表边框
            pygame.draw.rect(surface, (100, 100, 120), event_list_rect, 1)
            
        # 绘制当前悬停元素的工具提示
        if self.hover_element and self.hover_element in self.ui_elements:
            element = self.ui_elements[self.hover_element]
            if "tooltip" in element:
                mouse_pos = pygame.mouse.get_pos()
                draw_tooltip(surface, self.font, element["tooltip"], mouse_pos[0] + 15, mouse_pos[1] + 15)

# --- 神经网络可视化函数（与模拟器一致）---
def draw_neural_network(surface, font, agent, x, y, width, height, mouse_pos):
    """绘制神经网络可视化（与模拟器一致）"""
    title = font.render("计算核心 (Cᵢ) 拓扑图:", True, (200, 200, 100))
    surface.blit(title, (x, y))
    y += 30
    
    # 获取节点数量
    n_in = agent.gene.get('n_input', 0)
    n_hid = agent.gene.get('n_hidden', 0)
    n_out = agent.gene.get('n_output', 0)
    
    # 从基因中获取节点类型信息
    node_types = agent.gene.get('node_types', {
        'input': ['env_sense'] * n_in,
        'output': ['movement'] * 2 + ['signal'] * (n_out - 2) if n_out > 2 else ['movement'] * n_out,
        'hidden': ['standard'] * n_hid
    })
    
    # 为所有节点类型创建标签
    input_labels = []
    basic_input_labels = ["N_v", "N_gx", "N_gy", "H_v", "H_gx", "H_gy", "B1_v", "B1_gx", "B1_gy", "B2_v", "B2_gx", "B2_gy"]
    signal_in_count = 0
    for i in range(n_in):
        node_type = None
        if hasattr(agent, 'gene') and 'node_types' in agent.gene and 'input' in agent.gene['node_types'] and i < len(agent.gene['node_types']['input']):
            node_type = agent.gene['node_types']['input'][i]
        if node_type == 'signal_sense':
            input_labels.append(f"SigIn_{signal_in_count+1}")
            signal_in_count += 1
        elif i < len(basic_input_labels):
            input_labels.append(basic_input_labels[i])
        else:
            input_labels.append(f"In_{i}")
    output_labels = []
    if n_out > 0:
        output_labels.append("MoveX")
    if n_out > 1:
        output_labels.append("MoveY")
    for i in range(2, n_out):
        output_labels.append(f"Sig_{i-1}")
    col_x = [x + 30, x + width // 2, x + width - 30]
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
    # 绘制连接
    for from_n, to_n, weight in agent.gene.get('connections', []):
        if from_n in node_positions and to_n in node_positions:
            start_pos, end_pos = node_positions[from_n], node_positions[to_n]
            line_width = min(3, max(1, abs(int(weight * 2))))
            color = (0, min(255, 100 + int(abs(weight) * 80)), 0) if weight > 0 else (min(255, 150 + int(abs(weight) * 50)), 50, 50)
            pygame.draw.line(surface, color, start_pos, end_pos, line_width)
    hover_info = None
    for node_id, pos in node_positions.items():
        is_input = node_id < n_in
        is_hidden = n_in <= node_id < n_in + n_hid
        is_output = node_id >= n_in + n_hid
        if is_input:
            if node_id < len(basic_input_labels):
                color = (100, 100, 255)
            else:
                color = (180, 100, 255)
        elif is_hidden:
            color = (255, 165, 0)
        else:
            output_idx = node_id - (n_in + n_hid)
            if output_idx < 2:
                color = (255, 255, 100)
            else:
                color = (100, 255, 100)
        radius = 6
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, (0,0,0), pos, radius, 1)
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
        if math.hypot(mouse_pos[0] - pos[0], mouse_pos[1] - pos[1]) < radius:
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
            hover_info = (f"{node_type}节点 {node_id}", "", mouse_pos)
    if hover_info:
        title, _, pos = hover_info
        title_surf = font.render(title, True, (255, 255, 255))
        box_rect = pygame.Rect(pos[0] + 10, pos[1] + 10, title_surf.get_width() + 20, 30)
        pygame.draw.rect(surface, (0,0,0,200), box_rect)
        surface.blit(title_surf, (box_rect.x + 10, box_rect.y + 5))

# --- 工具提示辅助函数 ---
def draw_tooltip(surface, font, text, x, y, max_width=300):
    """绘制工具提示"""
    # 分割文本以适应最大宽度
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if font.size(test_line)[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:  # 确保当前行不为空
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # 处理单词比最大宽度还长的情况
                lines.append(word)
                current_line = []
    
    if current_line:  # 添加最后一行
        lines.append(' '.join(current_line))
    
    # 绘制背景
    line_height = font.get_height()
    box_height = len(lines) * line_height + 10
    box_width = min(max_width, max([font.size(line)[0] for line in lines])) + 20
    
    # 确保提示框不会超出屏幕边界
    if x + box_width > surface.get_width():
        x = surface.get_width() - box_width - 5
    if y + box_height > surface.get_height():
        y = surface.get_height() - box_height - 5
    
    box_rect = pygame.Rect(x, y, box_width, box_height)
    pygame.draw.rect(surface, TOOLTIP_COLOR, box_rect, border_radius=5)
    pygame.draw.rect(surface, (150, 150, 150), box_rect, 1, border_radius=5)
    
    # 绘制文本
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, (255, 255, 255))
        surface.blit(text_surface, (x + 10, y + 5 + i * line_height))

def is_point_in_rect(point, rect):
    """检查点是否在矩形内"""
    return rect.x <= point[0] <= rect.x + rect.width and rect.y <= point[1] <= rect.y + rect.height

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
    controller.frame_advance_time = 0
    controller.playback_rate = 1.0  # 默认播放速率
    
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