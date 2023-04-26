import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from dataclasses import dataclass
from enum import Enum
from queues import FIFOQueue, LIFOStack, PriorityQueue

WIDTH = 800
N_SIDE = 15
SLEEP = 10 # ms
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

TRIANGLE = np.array([(0,1), (0.5,0), (1,1)])
SMALL_TRIANGLE = np.array([(0.33, 0.33),(0.5, 0),(0.66, 0.33)])
# SMALL_TRIANGLE = np.array([(0.33, 1), (0.5, 0.66), (0.66, 1)])
SQUARE = np.array([(0,0), (0,1), (1,1), (1,0)])

Actions = Enum('Actions', ['FORWARD', 'ROTATE'])
action_cost = {
    Actions.FORWARD: 1,
    Actions.ROTATE: 0
}
MapValues = Enum('MapValues', ['EMPTY', 'BARRIER', 'FRONTIER', 'CLOSED'])

class Node:
    def __init__(self, player, actions=None, g=0, h=0):
        self.player = player
        self.actions = actions if actions is not None else []
        self.g = g
        self.h = h
        
    @property
    def cost(self):
        return self.g + self.h
        
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Node):
            return hash(self) == hash(other)
        return NotImplemented
    
    def __hash__(self) -> int:
        return hash(self.player)


@ dataclass
class Player:
    row: int = None
    col: int = None
    ori: int = 0
    
    def __hash__(self) -> int:
        return hash((self.row, self.col, self.ori))
    
    def copy(self):
        return Player(self.row, self.col, self.ori)
        
    def clear(self):
        self.row = None
        self.col = None
        self.ori = 0
        
    def set(self, row, col, ori):
        self.row = row
        self.col = col
        self.ori = ori
    
    def next_ori(self):        
        return (self.ori + 1) % 4
    
    def rotate(self):
        self.ori = self.next_ori()
    
    def is_set(self):
        return not (self.row is None or self.col is None)
    
    def next_pos(self):
        row, col, ori = self.row, self.col, self.ori
        
        if ori == 0:
            row -= 1
        elif ori == 1:
            col += 1
        elif ori == 2:
            row += 1
        elif ori == 3:
            col -= 1
            
        return row, col
    
    def act(self, action):
        if action == Actions.FORWARD:
            self.row, self.col = self.next_pos()
        elif action == Actions.ROTATE:
            self.rotate()
            
    
@dataclass
class Target:
    row: int = None
    col: int = None
    
    def is_set(self):
        return not (self.row is None or self.col is None)
    def clear(self):
        self.row = None
        self.col = None


def manhattan_distance(player, target):
        return abs(player.row - target.row) + abs(player.col - target.col)  

Methods = Enum('Methods', ['BFS', 'DFS', 'Dijkstra', 'AStar'])

class Game:
    def __init__(self, window, real_width, n_side, method_name):
        self.window = window
        self.N = n_side
        self.width = real_width
        self.player = Player()
        self.target = Target()
        
        self.map = self.empty_map(n_side)
        
        self.plan = [] 
        self.update_method(method_name)
        
        self.open = 0
        self.closed = 0
        
    
    def update_method(self, method_name):
        self.method_name = method_name
        self.path_finding_method = {
            Methods.BFS: self.breadth_first_search,
            Methods.DFS: self.depth_first_search,
            Methods.Dijkstra: self.uniform_cost_search,
            Methods.AStar: self.a_star_search
        }[method_name]
        
    @staticmethod
    def empty_map(n_side):
        return np.ones((n_side, n_side)) * MapValues.EMPTY.value
        
    def draw_grid(self):
        win = self.window
        width = self.width
        gap = self.width / self.N
        
        for i in range(self.N):
            pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
            for j in range(self.N):
                pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))
    
    def draw_poly(self, poly, row, col, color, rot=0):
        w = self.width / self.N
        h = self.width / self.N
        x,y = col * w, row * h
        win = self.window
        
        points = poly
        if rot != 0:
            points = np.array(points)-0.5
            rot_a = np.radians(rot)
            rotation_matrix = np.array([
                [np.cos(rot_a), -np.sin(rot_a), 0],
                [np.sin(rot_a), np.cos(rot_a), 0],
                [0, 0, 1]
            ])
            points = np.dot(
                rotation_matrix, 
                np.concatenate((points, np.ones((3,1))), axis=1).T).T[:,:2] + 0.5
        points = points * np.array([w,h]) + np.array([x,y])
        pygame.draw.polygon(win, color,points)
        
        
    def draw_player(self):
        if self.player is None or not self.player.is_set():
            return
        poly = TRIANGLE
        self.draw_poly(
            poly, row = self.player.row, col = self.player.col,
            color = RED, rot = self.player.ori * 90
        )
        
    def draw_plan(self):
        player = self.player.copy()
        for act in self.plan:
            player.act(act)
            self.draw_poly(SMALL_TRIANGLE, player.row, player.col, ORANGE, player.ori * 90)
        
    def draw_blocks(self):
        for row in range(self.N):
            for col in range(self.N):
                color = WHITE
                if self.map[row, col] == MapValues.BARRIER.value:
                    color = BLACK
                elif self.map[row, col] == MapValues.FRONTIER.value:
                    color = GREEN
                elif self.map[row, col] == MapValues.CLOSED.value:
                    color = BLUE
                
                self.draw_poly(SQUARE, row, col, color)
                    
    def draw_target(self):
        if not self.target.is_set():
            return
        self.draw_poly(SQUARE, self.target.row, self.target.col, PURPLE)
    
        
    def draw(self):
        win = self.window
        # Draw background
        win.fill(WHITE)
        # Draw blocks
        self.draw_blocks()
        # Draw target
        self.draw_target()
        # Draw player
        self.draw_player()
        # Draw plan
        self.draw_plan()
        # Draw grid
        self.draw_grid()
        
        # Write text
        pygame.font.init()
        my_font = pygame.font.Font('./fonts/roboto.ttf', 40)
        text_surface = my_font.render(self.method_name.name, False, (0, 0, 0))
        win.blit(text_surface, dest=(5,5))
        text_surface = my_font.render(f'C:{self.closed} F:{self.open}', False, (0, 0, 0))
        win.blit(text_surface, dest=(5,45))
        
        pygame.display.update()
        
    def get_clicked_pos(self, pos):
        gap = self.width / self.N
        x, y = pos
        row = y // gap
        col = x // gap
        return int(row), int(col)
    
    def add_block(self, row, col):
        self.map[row, col] = MapValues.BARRIER.value
        
    def clear_at(self, row, col):
        # clear player
        if self.player.row == row and self.player.col == col:
            self.player.clear()
        # clear target
        if self.target.row == row and self.target.col == col:
            self.target.clear()
        # clear block
        self.map[row, col] = 0   
        
    def is_valid_pos(self, row, col):
        return row >= 0 and row < self.N and col >= 0 and col < self.N  and self.map[row, col] != MapValues.BARRIER.value
    
    def clear(self):
        self.player.clear()
        self.target.clear()
        self.map = self.empty_map(self.N)
        self.plan = []
        
        self.open = 0
        self.closed = 0
        
    def update_plan_map(self, frontier, closed):
        for node in frontier:
            row, col = node.player.row, node.player.col
            if self.map[row, col] == MapValues.EMPTY.value: 
                self.map[row, col] = MapValues.FRONTIER.value
        for node in closed:
            row, col = node.player.row, node.player.col
            if self.map[row, col] == MapValues.EMPTY.value or self.map[row, col] == MapValues.FRONTIER.value: 
                self.map[row, col] = MapValues.CLOSED.value
    
    def tidy_map(self):
        self.map[self.map == MapValues.FRONTIER.value] = MapValues.EMPTY.value
        self.map[self.map == MapValues.CLOSED.value] = MapValues.EMPTY.value
                
    def make_plan(self):
        self.tidy_map()
        self.plan = []
        def callback(current,frontier, closed):
            self.update_plan_map(frontier, closed)
            self.open = len(frontier)
            self.closed = len(closed)
            self.draw()
            self.draw_poly(SMALL_TRIANGLE, current.player.row, current.player.col, WHITE, current.player.ori * 90)
            pygame.display.update()
            pygame.time.delay(SLEEP)
        self.plan_algorithm(callback)
        
    def plan_algorithm(self,callback):
        # self.breadth_first_search(callback)
        # self.depth_first_search(callback)
        # self.uniform_cost_search(callback)
        # self.a_star_search(callback)
        self.path_finding_method(callback)
    
    def breadth_first_search(self,callback):
        frontier = FIFOQueue()
        current = Node(player=self.player.copy(),actions=[])
        frontier.push(current)
        closed = set()
        plan = []
        while len(frontier) > 0:
            current = frontier.pop()
            # if current in closed continue
            if current in closed:
                continue
            # if current is target return actions
            if current.player.row == self.target.row and current.player.col == self.target.col:
                plan = current.actions
                break
            # add current to closed
            closed.add(current)
            # Expand children
            for action in Actions:
                child_player = current.player.copy()
                child_player.act(action)
                if self.is_valid_pos(child_player.row, child_player.col):
                    child_node = Node(
                        child_player,
                        actions = current.actions + [action]
                    )
                    frontier.push(child_node)
            # Update callback
            callback(current, frontier, closed)
        self.plan = plan
        
    def depth_first_search(self, callback):
        frontier = LIFOStack()
        current = Node(player=self.player.copy(),actions=[])
        frontier.push(current)
        closed = set()
        plan = []
        while len(frontier) > 0:
            current = frontier.pop()
            # if current in closed continue
            if current in closed:
                continue
            # if current is target return actions
            if current.player.row == self.target.row and current.player.col == self.target.col:
                plan = current.actions
                break
            # add current to closed
            closed.add(current)
            # Expand children
            for action in list(Actions)[::-1]:
                child_player = current.player.copy()
                child_player.act(action)
                if self.is_valid_pos(child_player.row, child_player.col):
                    child_node = Node(
                        child_player,
                        actions = current.actions + [action]
                    )
                    frontier.push(child_node)
            # Update callback
            callback(current, frontier, closed)
        self.plan = plan
    
    def uniform_cost_search(self,callback):
        frontier = PriorityQueue()
        current = Node(player=self.player.copy(),actions=[],g=0)
        frontier.push(current, current.cost)
        closed = set()
        plan = []
        while len(frontier) > 0:
            current = frontier.pop()
            # if current in closed continue
            if current in closed:
                continue
            # if current is target return actions
            if current.player.row == self.target.row and current.player.col == self.target.col:
                plan = current.actions
                break
            # add current to closed
            closed.add(current)
            # Expand children
            for action in Actions:
                child_player = current.player.copy()
                child_player.act(action)
                if self.is_valid_pos(child_player.row, child_player.col):
                    child_node = Node(
                        child_player,
                        actions = current.actions + [action],
                        g=current.cost + action_cost[action]
                    )
                    frontier.push(child_node, child_node.cost)
            # Update callback
            callback(current, frontier, closed)
        self.plan = plan
    
    def a_star_search(self,callback):
        frontier = PriorityQueue()
        current = Node(player=self.player.copy(),actions=[],g=0, h=manhattan_distance(self.player, self.target))
        frontier.push(current, current.cost)
        closed = set()
        plan = []
        while len(frontier) > 0:
            current = frontier.pop()
            # if current in closed continue
            if current in closed:
                continue
            # if current is target return actions
            if current.player.row == self.target.row and current.player.col == self.target.col:
                plan = current.actions
                break
            # add current to closed
            closed.add(current)
            # Expand children
            for action in Actions:
                child_player = current.player.copy()
                child_player.act(action)
                if self.is_valid_pos(child_player.row, child_player.col):
                    g = current.g + action_cost[action]
                    h = manhattan_distance(child_player, self.target)
                    child_node = Node(
                        child_player,
                        actions = current.actions + [action],
                        g=g, h=h
                    )
                    frontier.push(child_node, child_node.cost)
            # Update callback
            callback(current, frontier, closed)
        self.plan = plan

if __name__ == "__main__":
    game = Game(WIN, WIDTH, N_SIDE, method_name=Methods.BFS)
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # Mouse events 
            pos = pygame.mouse.get_pos()
            row, col = game.get_clicked_pos(pos)    
            if pygame.mouse.get_pressed()[0]: # LEFT
                if not game.player.is_set():
                    game.player.set(row, col, 0)
                elif not game.target.is_set():
                    # if different position
                    if row != game.player.row or col != game.player.col:
                        game.target = Target(row=row, col=col)
                else:
                    # if different position
                    if (row != game.player.row or col != game.player.col) and (row != game.target.row or col != game.target.col):
                        game.add_block(row, col)
                    
            elif pygame.mouse.get_pressed()[2]: # RIGHT
                game.clear_at(row, col)
            # Key events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    game.clear()
                if event.key == pygame.K_r:
                    if game.player.is_set():
                        game.player.rotate()
                if event.key == pygame.K_f:
                    if game.player.is_set():
                        game.player.act(Actions.FORWARD)
                if event.key == pygame.K_SPACE:
                    plan = game.make_plan()
                if event.key == pygame.K_1:
                    game.update_method(Methods.BFS)
                if event.key == pygame.K_2:
                    game.update_method(Methods.DFS)
                if event.key == pygame.K_3:
                    game.update_method(Methods.Dijkstra)
                if event.key == pygame.K_4:
                    game.update_method(Methods.AStar)
        game.draw()
        