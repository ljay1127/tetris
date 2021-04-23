import numpy as np
import pygame
import sys
from collections import deque
import random
import os.path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

# TETRIS ENVIRONMENT

class Tetris:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((24,10))

        # SAR
        self.state = deque(maxlen=4)
        self.state.append(self.grid.copy())
        self.state.append(self.grid.copy())
        self.state.append(self.grid.copy())
        self.state.append(self.grid.copy())
        self.action = 0
        self.reward = 0
        self.done = False

        self.BLOCK_SIZE = 30

        # PIECE INFO
        self.piece = 0
        self.piece_pos = 0

        # GAME INFO
        self.n_holes = 0
        self.curr_height = 23

        return self.state, self.action, self.reward, self.done
    
    def render(self):
        for i in range(20):
            for j in range(10):
                if self.grid[i + 4, j] == 1:
                    tmp_rec = pygame.Rect(j * self.BLOCK_SIZE, i * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 0, 255), tmp_rec)
                elif self.grid[i + 4, j] == 2:
                    tmp_rec = pygame.Rect(j * self.BLOCK_SIZE, i * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 255, 0), tmp_rec)
                else:
                    tmp_rec = pygame.Rect(j * self.BLOCK_SIZE, i * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 0, 0), tmp_rec)

    def step(self, action):
        self.action = action
        
        if self.action == 0:
            self.reward = 0
            self.drop_down()
        elif self.action == 1:
            self.reward = -0.1
            self.move_left()
        elif self.action == 2:
            self.reward = -0.1
            self.move_right()
        elif self.action == 3:
            self.reward = 0
            self.drop_down()
        elif self.action == 4:
            self.reward = -0.1
            self.rotate_piece()

        self.state.append(self.grid.copy())
        self.done = self.check_game_over()

        return self.state, self.action, self.reward, self.done

    def create_piece(self):
        r_num = random.randint(0, 6)

        ##
        ##
        if r_num == 0:
            self.grid[3:5, 4:6] = 2
         #
        ###
        elif r_num == 1:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 3] = 0
            self.grid[3, 5] = 0
        ####
        elif r_num == 2:
            self.grid[3, 3:7] = 2 
         ##
        ##
        elif r_num == 3:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 3] = 0
            self.grid[4, 5] = 0
        ##
         ##
        elif r_num == 4:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 5] = 0
            self.grid[4, 3] = 0
          #
        ###
        elif r_num == 5:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 3:5] = 0
        #
        ###
        elif r_num == 6:
            self.grid[3:5, 3:6] = 2
            self.grid[3, 4:6] = 0

        self.piece = r_num
        self.piece_pos = 0

    def get_active_piece(self):
        active_piece = []
        for i in range(24):
            for j in range(10):
                if self.grid[i,j] == 2:
                    active_piece.append([i,j])
        return active_piece

    def check_collision(self, new_location):
        for i in new_location:
            if i[0] > 23:
                return True
            elif i[0] < 0:
                return True
            elif i[1] > 9:
                return True
            elif i[1] < 0:
                return True
            elif self.grid[i[0], i[1]] == 1:
                return True
        return False

    def update_piece_location(self, active_piece, new_location):
        for i in active_piece:
            self.grid[i[0], i[1]] = 0
        for i in new_location:
            self.grid[i[0], i[1]] = 2

    def change_piece_status(self, active_piece):
        for i in active_piece:
            self.grid[i[0], i[1]] = 0
        for i in active_piece:
            self.grid[i[0], i[1]] = 1

    def check_game_over(self):
        for i in range(24):
            for j in range(10):
                if self.grid[i,j] == 1 and i < 4:
                #if self.grid[i,j] == 1 and i < 18:
                    self.reward = -100
                    return True
        return False

    def drop_down(self):
        active_piece = self.get_active_piece()
        new_location = []
        for i in active_piece:
            new_location.append([i[0] + 1, i[1]])
        if not self.check_collision(new_location):
            self.update_piece_location(active_piece, new_location)
        else:
            self.change_piece_status(active_piece)
            self.check_grid()

    def move_left(self):
        active_piece = self.get_active_piece()
        new_location = []
        for i in active_piece:
            new_location.append([i[0], i[1] - 1])
        if not self.check_collision(new_location):
            self.update_piece_location(active_piece, new_location)

    def move_right(self):
        active_piece = self.get_active_piece()
        new_location = []
        for i in active_piece:
            new_location.append([i[0], i[1] + 1])
        if not self.check_collision(new_location):
            self.update_piece_location(active_piece, new_location)

    def rotate_piece(self):
        active_piece = self.get_active_piece()
        new_location = []
         #
        ###
        if self.piece == 1:
             #
            ###
            if self.piece_pos == 0:
                row, col = active_piece[1]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            ##
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row - 1, col + 2])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 2
            ###
             #
            elif self.piece_pos == 2:
                row, col = active_piece[-1]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                new_location.append([row - 1, col - 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 3
             #
            ##
             #
            elif self.piece_pos == 3:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row - 1, col])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
        ####
        elif self.piece == 2:
            ####
            if self.piece_pos == 0:
                row, col = active_piece[0]
                new_location.append([row - 1, col + 1])
                new_location.append([row - 2, col + 1])
                new_location.append([row - 3, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            #
            #
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row, col + 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
         ##
        ##
        elif self.piece == 3:
             ##
            ##
            if self.piece_pos == 0:
                row, col = active_piece[-2]
                new_location.append([row - 2, col])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            ##
             #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
        ##
         ##
        elif self.piece == 4:
            ##
             ##
            if self.piece_pos == 0:
                row, col = active_piece[-2]
                new_location.append([row, col - 1])
                new_location.append([row - 1, col - 1])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
             #
            ##
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col + 1])
                new_location.append([row, col + 2])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
          #
        ###
        elif self.piece == 5:
              #
            ###
            if self.piece_pos == 0:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col - 2])
                new_location.append([row - 1, col - 2])
                new_location.append([row - 2, col - 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            #
            #
            ##
            elif self.piece_pos == 1:
                row, col = active_piece[-2]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 1, col + 1])
                new_location.append([row - 1, col + 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 2
            ###
            #
            elif self.piece_pos == 2:
                row, col = active_piece[-1]
                new_location.append([row - 2, col])
                new_location.append([row - 2, col + 1])
                new_location.append([row - 1, col + 1])
                new_location.append([row, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 3
            ##
             #
             #
            elif self.piece_pos == 3:
                row, col = active_piece[-1]
                new_location.append([row, col - 1])
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row - 1, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0
        #
        ###
        elif self.piece == 6:
            #
            ###
            if self.piece_pos == 0:
                row, col = active_piece[1]
                new_location.append([row, col])
                new_location.append([row - 1, col])
                new_location.append([row - 2, col])
                new_location.append([row - 2, col + 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 1
            ##
            #
            #
            elif self.piece_pos == 1:
                row, col = active_piece[-1]
                new_location.append([row, col + 2])
                new_location.append([row-1, col])
                new_location.append([row-1, col + 1])
                new_location.append([row-1, col + 2])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 2
            ###
              #
            elif self.piece_pos == 2:
                row, col = active_piece[-1]
                new_location.append([row, col - 2])
                new_location.append([row, col - 1])
                new_location.append([row - 1, col - 1])
                new_location.append([row - 2, col - 1])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 3
             #
             #
            ##
            elif self.piece_pos == 3:
                row, col = active_piece[-2]
                new_location.append([row, col])
                new_location.append([row, col + 1])
                new_location.append([row, col + 2])
                new_location.append([row - 1, col])
                if not self.check_collision(new_location):
                    self.update_piece_location(active_piece, new_location)
                    self.piece_pos = 0

    def check_grid(self):
        # CHECK FOR HOLES ON THE GAME
        hole_found = 0
        height = 0
        for i in range(23):
            for j in range(10):
                if self.grid[i, j] == 1:
                    if height == 0:
                        height = i
                    if self.grid[i + 1, j] == 0:
                        hole_found += 1
        # if hole_found > self.n_holes:
        #     self.reward = -10
        # elif hole_found < self.n_holes:
        #     self.reward = 10
        if height != self.curr_height:
            self.reward = height - 20
        else:
            self.reward = 1
        self.curr_height = height
        # elif hole_found == self.n_holes:
        #     test = self.get_active_piece()
        #     if len(test) == 0:
        #         self.reward = 1
        self.n_holes = hole_found

        # CHECK FOR ROW TO REMOVE
        delete_this_row = []
        for i in range(24):
            delete_row = True
            for j in range(10):
                if self.grid[i, j] == 0:
                    delete_row = False
                    break
            if delete_row:
                delete_this_row.append(i)
        
        if len(delete_this_row) > 0:
            self.curr_height += len(delete_this_row)
            self.reward = 10 * len(delete_this_row)

            new_grid = self.grid.copy()
            new_grid = np.delete(new_grid, delete_this_row, axis=0)
            self.grid = np.zeros((24, 10), dtype=int)
            self.grid[24 - new_grid.shape[0]: ,:] = new_grid

# DQN AGENT
class DQN_Agent:
    def __init__(self):
        self.n_actions = 4

        self.rng = 1
        self.rng_decay = 0.995
        self.rng_min = 0.01
        self.discount = 0.95

        self.memory = deque(maxlen=50_000)
        self.train_ctr = 0
        self.transfer_ctr = 0

        self.q_eval = self.create_model()
        self.q_target = self.create_model()
        self.transfer_model()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(64, 3, 2, activation='relu', padding='same', input_shape=(24,10,4)))
        model.add(MaxPool2D(pool_size=2, strides=1, padding='same'))
        model.add(Conv2D(128, 2, 1, activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=1, padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(5, activation='linear'))
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        return model

    def train(self):
        if len(self.memory) < 10_000:
            return

        # self.train_ctr += 1

        # if self.train_ctr < 1_000:
        #     return
        
        # self.train_ctr = 0

        mini_batch = random.sample(self.memory, 32)

        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs = self.q_eval.predict(current_states)
        next_states = np.array([transition[3] for transition in mini_batch])
        next_qs = self.q_target.predict(next_states)

        X = current_states
        y =[]

        for i, observation in enumerate(mini_batch):
            _ = observation[0]
            action = observation[1]
            reward = observation[2]
            new_state = observation[3]
            done = observation[4]

            if done:
                new_q = reward
            else:
                new_q = reward + self.discount * np.max(next_qs[i])

            qs = current_qs[i]
            qs[action] = new_q
            y.append(qs)

        self.q_eval.fit(X, np.array(y), verbose=0, shuffle=False)

        self.transfer_ctr += 1
        if self.transfer_ctr > 5:
            self.transfer_model()
            self.transfer_ctr = 0

            self.rng = self.rng * self.rng_decay
            if self.rng < self.rng_min:
                #self.rng = self.rng_min
                self.rng = 0.3
            print(self.rng)


    def get_q(self, state):
        if random.random() < self.rng:
            return random.randint(0,4)
        else:
            state = self.reshape_state(state)
            qs = self.q_eval.predict(np.reshape(state, (1,24,10,4)))
            return np.argmax(qs)

    def transfer_model(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def remember(self, current_state, action, reward, next_state, done):
        current_state = self.reshape_state(current_state)
        next_state = self.reshape_state(next_state)
        self.memory.append([current_state, action, reward, next_state, done])

    def save(self):
        self.q_eval.save('tetris.model')

    def load(self):
        self.q_eval = tf.keras.models.load_model('tetris.model')
        self.transfer_model()
        self.rng = 0.01

    def reshape_state(self, state):
        frame = []
        for i in state:
            frame.append(np.reshape(np.array(i), (24,10,1)))
        frame = np.array(frame)

        new_frame = np.zeros((24,10,4))

        new_frame[:,:,0] = frame[0,:,:,0]
        new_frame[:,:,1] = frame[1,:,:,0]
        new_frame[:,:,2] = frame[2,:,:,0]
        new_frame[:,:,3] = frame[3,:,:,0]

        return new_frame


# MAIN GAME LOOP

# SCREEN
pygame.init()
SCREEN_SIZE = width, height = 300, 600
screen = pygame.display.set_mode(SCREEN_SIZE)

# DELTA TIME
current_time = pygame.time.get_ticks()
previous_time = current_time
delta_time = current_time - previous_time
drop_time = 0
control_time = 0

# INITIALIZE
game = Tetris()
agent = DQN_Agent()
if os.path.exists('tetris.model'):
    agent.load()
    print('agent loaded!')
else:
    print('no saved agent found!')
state, action, reward, done = game.reset()
current_state = state
next_state = state
ai_player = True
paused = False

while True:
    # UPDATE DELTA TIME
    current_time = pygame.time.get_ticks()
    delta_time = current_time - previous_time
    previous_time = current_time

    if paused:
        delta_time = 0

    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            agent.save()
            sys.exit()
        # KEYBOARD EVENTS
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 4
            elif event.key == pygame.K_ESCAPE:
                agent.save()
                sys.exit()
            elif event.key == pygame.K_SPACE:
                if ai_player == True:
                    ai_player = False
                else:
                    ai_player = True
            elif event.key == pygame.K_p:
                if paused:
                    paused = False
                else:
                    paused = True

    # KEYBOARD LONG PRESS
    keys = pygame.key.get_pressed()
    if keys[pygame.K_DOWN]:
        action = 3
    elif keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2

    # NO MORE ACTIVE PIECE
    active_piece = game.get_active_piece()
    if len(active_piece) == 0:
        game.create_piece()

    if ai_player == True:
        # CONTROLS
        control_time += delta_time
        if control_time > 10:
            # AI
            state, action, reward, done = game.step(agent.get_q(current_state))
            #state, action, reward, done = game.step(action)
            next_state = state
            # AI
            agent.remember(current_state, action, reward, next_state, done)
            current_state = next_state
            control_time = 0

        # GRAVITY DROP
        drop_time += delta_time
        if drop_time > 10:
            state, action, reward, done = game.step(0)
            next_state = state
            # AI
            agent.remember(current_state, action, reward, next_state, done)
            current_state = next_state
            drop_time = 0
    else:
        # CONTROLS
        control_time += delta_time
        if control_time > 75 and action != 0:
            state, action, reward, done = game.step(action)
            next_state = state
            # AI
            agent.remember(current_state, action, reward, next_state, done)
            current_state = next_state
            control_time = 0

        # GRAVITY DROP
        drop_time += delta_time
        if drop_time > 500:
            state, action, reward, done = game.step(0)
            next_state = state
            # AI
            agent.remember(current_state, action, reward, next_state, done)
            current_state = next_state
            drop_time = 0

    # SCREEN DRAW
    screen.fill((0, 0, 0))
    game.render()
    pygame.display.flip()

    # GAME OVER
    if done:
        agent.train()
        state, action, reward, done = game.reset()
        next_state = state
        current_state = next_state