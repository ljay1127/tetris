import sys
import pygame
import numpy as np
import random

# CONSTANT VARIABLES
# Appearance variables
BLOCK_SIZE = 30
SCREEN_SIZE = width, height = 300, 600
# Time related variables
TIME_KICK = 500
delta = 0
down_delta = 0
direction_delta = 0
current_delta = 0
last_delta = 0
elapsed_delta = 0
# Active piece variables
current_piece = 0
current_piece_status = 0
active_block = []
offset = 0

# PYGAME INITIALIZATION
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)

# NUMPY ARRAY OF THE GAMING LANDSCAPE
grid = np.zeros((20, 10), dtype=int)

# GET DELTA TIME
def get_delta_time(last_delta):
    current_delta = pygame.time.get_ticks()
    elapsed_delta = current_delta - last_delta
    last_delta = current_delta
    return elapsed_delta, last_delta

# RANDOM PIECE GENERATOR
def random_piece():
    offset = 0
    r_num = random.randint(0, 6)
    if r_num == 0:
        piece = np.full((2,2), fill_value=2, dtype=int)
        offset = 4
    elif r_num == 1:
        piece = np.full((2,3), fill_value=2, dtype=int)
        piece[0, 0] = 0
        piece[0, 2] = 0
        offset = 4
    elif r_num == 2:
        piece = np.full((1,4), fill_value=2, dtype=int)
        offset = 3
    elif r_num == 3:
        piece = np.full((2,3), fill_value=2, dtype=int)
        piece[0, 2] = 0
        piece[1, 0] = 0
        offset = 4
    elif r_num == 4:
        piece = np.full((2,3), fill_value=2, dtype=int)
        piece[0, 0] = 0
        piece[1, 2] = 0
        offset = 4
    elif r_num == 5:
        piece = np.full((2,3), fill_value=2, dtype=int)
        piece[0, 1] = 0
        piece[0, 2] = 0
        offset = 4
    else:
        piece = np.full((2,3), fill_value=2, dtype=int)
        piece[0, 1] = 0
        piece[0, 0] = 0
        offset = 3
    return piece, r_num, offset, 0

# CHECK ROTATE COLLISION
def check_collision(new_shape):
    no_collision = True
    for i in new_shape:
        if i[0] < 0:
            no_collision = False
        if i[0] > 19:
            no_collision = False
        if i[1] < 0:
            no_collision = False
        if i[1] > 9:
            no_collision = False
        if no_collision:
            if grid[i[0], i[1]] == 1:
                no_collision = False
    return no_collision

# ROTATE PIECE
def rotate_piece():
    #
    #    #
    #   ###    ROTATING THIS PIECE
    #
    global current_piece, current_piece_status
    if current_piece == 1:
        if current_piece_status == 0:
            bottom_left = active_block[1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 2))
            new_shape.append((bottom_left[0] - 2, bottom_left[1] + 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 1
        elif current_piece_status == 1:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] - 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 2
        elif current_piece_status == 2:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] - 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 2, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 3
        elif current_piece_status == 3:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0], bottom_left[1] - 1))
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 0
    #
    #    ####    ROTATING THIS PIECE
    #
    elif current_piece == 2:
        if current_piece_status == 0:
            bottom_left = active_block[0]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 2, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 3, bottom_left[1] + 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 1
        elif current_piece_status == 1:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] - 1))
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0], bottom_left[1] + 2))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 0
    #
    #   ##
    #    ##    ROTATING THIS PIECE
    #
    elif current_piece == 3:
        if current_piece_status == 0:
            bottom_left = active_block[-2]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] - 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] - 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 2, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 1
        elif current_piece_status == 1:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0], bottom_left[1] + 2))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 0
    #
    #    ##
    #   ##     ROTATING THIS PIECE
    #
    elif current_piece == 4:
        if current_piece_status == 0:
            bottom_left = active_block[-2]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 2, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 1
        elif current_piece_status == 1:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] - 1))
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 0
    #
    #   #  
    #   ###    ROTATING THIS PIECE
    #
    elif current_piece == 5:
        if current_piece_status == 0:
            bottom_left = active_block[1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 2, bottom_left[1]))
            new_shape.append((bottom_left[0] - 2, bottom_left[1] + 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 1
        elif current_piece_status == 1:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] + 2))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 2))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 2
        elif current_piece_status == 2:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] - 2))
            new_shape.append((bottom_left[0], bottom_left[1] - 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] - 1))
            new_shape.append((bottom_left[0] - 2, bottom_left[1] - 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 3
        elif current_piece_status == 3:
            bottom_left = active_block[-2]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0], bottom_left[1] + 2))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 0
    #
    #     #
    #   ###    ROTATING THIS PIECE
    #
    elif current_piece == 6:
        if current_piece_status == 0:
            bottom_left = active_block[1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 2, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 1
        elif current_piece_status == 1:
            bottom_left = active_block[-2]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1]))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 2))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 2
        elif current_piece_status == 2:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 2, bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 2, bottom_left[1]))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 3
        elif current_piece_status == 3:
            bottom_left = active_block[-1]
            new_shape = []
            new_shape.append((bottom_left[0], bottom_left[1] - 1))
            new_shape.append((bottom_left[0], bottom_left[1]))
            new_shape.append((bottom_left[0], bottom_left[1] + 1))
            new_shape.append((bottom_left[0] - 1, bottom_left[1] + 1))
            if check_collision(new_shape):
                for i in active_block:
                    grid[i[0], i[1]] = 0
                for i in new_shape:
                    grid[i[0], i[1]] = 2
                current_piece_status = 0

# ACTIVE PIECE MOVE LEFT
def move_left():
    global grid, active_block
    new_shape = []
    for i in active_block:
        new_shape.append((i[0], i[1] - 1))
    if check_collision(new_shape):
        for i in active_block:
            grid[i[0], i[1]] = 0
        for i in new_shape:
            grid[i[0], i[1]] = 2

# ACTIVE PIECE MOVE RIGHT
def move_right():
    global grid, active_block
    new_shape = []
    for i in active_block:
        new_shape.append((i[0], i[1] + 1))
    if check_collision(new_shape):
        for i in active_block:
            grid[i[0], i[1]] = 0
        for i in new_shape:
            grid[i[0], i[1]] = 2

# ACTIVE PIECE MOVE DOWN
def move_down():
    global grid, active_block
    new_shape = []
    for i in active_block:
        new_shape.append((i[0] + 1, i[1]))
    if check_collision(new_shape):
        for i in active_block:
            grid[i[0], i[1]] = 0
        for i in new_shape:
            grid[i[0], i[1]] = 2

# CREATE PIECE
def create_piece():
    global piece, current_piece, current_piece_status
    piece, current_piece, offset, current_piece_status = random_piece()
    grid[0:piece.shape[0], offset:piece.shape[1] + offset] = piece

# DROP ACTIVE PIECE
def drop_active_piece():
    blocked = False

    for i in active_block:
        grid[i[0], i[1]] = 0
        if i[0] + 1 == 20:
            blocked = True
        else:
            if grid[i[0] + 1, i[1]] == 1:
                blocked = True
    if blocked:
        for i in active_block:
            grid[i[0], i[1]] = 1
    else:
        for i in active_block:
            grid[i[0] + 1, i[1]] = 2

# CONTROL FUNCTION
def get_keyboard_events(event):
    if event.key == pygame.K_UP:
        rotate_piece()
    # elif event.key == pygame.K_LEFT:
    #     move_left()
    # elif event.key == pygame.K_RIGHT:
    #     move_right()
    # elif event.key == pygame.K_DOWN:
    #     move_down()
    elif event.key == pygame.K_ESCAPE:
        sys.exit()

# CHECK FOR COMPLETED ROWS
def check_grid():
    global grid
    delete_this_row = []

    for i in range(20):
        delete_row = True
        for j in range(10):
            if grid[i, j] == 0:
                delete_row = False
        if delete_row:
            delete_this_row.append(i)

    if len(delete_this_row) > 0:
        new_grid = grid.copy()
        new_grid = np.delete(new_grid, delete_this_row, axis=0)
        grid = np.zeros((20, 10), dtype=int)
        grid[20 - new_grid.shape[0]: ,:] = new_grid


# MAIN GAME LOOP
game = True
while game:
    # QUIT GAME
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            get_keyboard_events(event)

    # DELTA TIME COMPUTE
    new_delta, last_delta = get_delta_time(last_delta)
    delta += new_delta
    down_delta += new_delta
    direction_delta += new_delta

    if down_delta > 50:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN]:
            move_down()
        down_delta = 0
    
    if direction_delta > 50:
        if keys[pygame.K_LEFT]:
            move_left()
        elif keys[pygame.K_RIGHT]:
            move_right()
        direction_delta = 0

    if delta > TIME_KICK:
        delta = 0
        # MOVE PIECE DOWN
        drop_active_piece()
        
       
    screen.fill((0, 0, 0))

    # DRAW BLOCKS
    active_block = []
    restart_flag = False
    for i in range(20):
        for j in range(10):
            if restart_flag == False:
                if grid[i, j] == 1:
                    if i == 0:
                        restart_flag = True
                    else:
                        tmp_rec = pygame.Rect(j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                        pygame.draw.rect(screen, (0, 255, 0), tmp_rec)
                elif grid[i, j] == 2:
                    active_block.append((i,j))
                    tmp_rec = pygame.Rect(j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 0, 255), tmp_rec)
                else:
                    tmp_rec = pygame.Rect(j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(screen, (0, 0, 0), tmp_rec)
    pygame.display.flip()

    if restart_flag:
        # NUMPY ARRAY OF THE GAMING LANDSCAPE
        grid = np.zeros((20, 10), dtype=int)

    # NO MORE ACTIVE PIECE. CREATE NEW ONE
    if len(active_block) == 0:
        check_grid()
        create_piece()