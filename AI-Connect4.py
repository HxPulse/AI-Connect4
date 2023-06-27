#%%

import random as rd
import numpy as np
from collections import defaultdict
import winsound
import pygame

#################################################################

# Tools we will be using in our algorithms and programs
# Some functions look the same but serve different purposes (reworked version of AlphaBeta used for MCTS) 

#################################################################

def game_over(grid):
    # Takes a grid
    # Returns 2 elements : 
    # - a boolean specifying wether the game is over or not
    # - the id of the winning player, otherwise "tie" or None if the game is still playable 
    rows = len(grid)
    cols = len(grid[0])

    for row in range(rows):
        for col in range(cols - 3):
            if grid[row][col] == grid[row][col+1] == grid[row][col+2] == grid[row][col+3] != 0:
                return True, grid[row][col] 

    for col in range(cols):
        for row in range(rows - 3):
            if grid[row][col] == grid[row+1][col] == grid[row+2][col] == grid[row+3][col] != 0:
                return True, grid[row][col]  

    for row in range(rows - 3):
        for col in range(cols - 3):
            if grid[row][col] == grid[row+1][col+1] == grid[row+2][col+2] == grid[row+3][col+3] != 0:
                return True, grid[row][col]  

    for row in range(rows - 3):
        for col in range(3, cols):
            if grid[row][col] == grid[row+1][col-1] == grid[row+2][col-2] == grid[row+3][col-3] != 0:
                return True, grid[row][col] 

    tiedGame = True
    for i in grid:
        for j in i:
            if j == 0:
                tiedGame = False
    if tiedGame:
        return True, "it's a tie"
    
    return False, None

def is_game_over(grid):
    return game_over(grid)[0]

def game_result(grid):
    # Takes a grid
    # Returns 1 for victory, -1 for a defeat and 0 for a tied game
    # Contrary to the previous functions, this one is only called if the game is sure to be over, when we want to know the winner using MCTS
    rows = len(grid)
    cols = len(grid[0])

    for row in range(rows):
        for col in range(cols - 3):
            if grid[row][col] == grid[row][col+1] == grid[row][col+2] == grid[row][col+3] == 1:
                return 1
            if grid[row][col] == grid[row][col+1] == grid[row][col+2] == grid[row][col+3] == 2:
                return -1

    for col in range(cols):
        for row in range(rows - 3):
            if grid[row][col] == grid[row+1][col] == grid[row+2][col] == grid[row+3][col] == 1:
                return 1
            if grid[row][col] == grid[row+1][col] == grid[row+2][col] == grid[row+3][col] == 2:
                return -1

    for row in range(rows - 3):
        for col in range(cols - 3):
            if grid[row][col] == grid[row+1][col+1] == grid[row+2][col+2] == grid[row+3][col+3] == 1:
                return 1
            if grid[row][col] == grid[row+1][col+1] == grid[row+2][col+2] == grid[row+3][col+3] == 2:
                return -1

    for row in range(rows - 3):
        for col in range(3, cols):
            if grid[row][col] == grid[row+1][col-1] == grid[row+2][col-2] == grid[row+3][col-3] == 1:
                return 1
            if grid[row][col] == grid[row+1][col-1] == grid[row+2][col-2] == grid[row+3][col-3] == 2:
                return -1

    tiedGame = True
    for i in grid:
        for j in i:
            if j == 0:
                tiedGame = False
    if tiedGame:
        return 0
    
def showGrid(grid):
    # Takes a grid
    # Prints the said grid in the command prompt
    for i in grid:
        print(i)
        
def makeMove(move, grid, val):
    # move = two integers array list representing the move to be played
    # grid = grid on which to play the move
    # val = player making the move (value)
    # Returns the new grid made after the move
    
    newgrid = [row[:] for row in grid]  # Creating a copy of the grid otherwise we are facing some issues
    newgrid[move[0]][move[1]] = val
    return newgrid     
  
def move(grid, move):
    # move = two integers array list representing the move to be played
    # grid = grid on which to play the move
    # Returns the new grid made after the move
    # Fonction légèrement différente de celle ci-dessus car modifiée pour être plus facilement implémentable dans MCTS
    # Slightly different function from the one above, provides an easier implementation into MCTS
    
    newgrid = [row[:] for row in grid]   # Creating a copy of the grid otherwise we are facing some issues
    newgrid[move[0]][move[1]] = move[2]
    return newgrid

def convertEvalIntoPattern(eval):
    # An evaluation is an array of 7 float that we use to fill our patterns
    # In our patterns, we look at all the combinations of a same evaluation. An eval can be seen as a compacted version of a pattern, removing the useless repetition
    
    p = [
        ([1, 1, 1, 1], eval[0]),
        ([2, 2, 2, 2], eval[1]),
        ([2, 2, 2, 1], eval[2]),
        ([2, 2, 1, 2], eval[2]),
        ([2, 1, 2, 2], eval[2]),
        ([1, 2, 2, 2], eval[2]),
        ([2, 2, 2, 0], eval[3]),
        ([2, 2, 0, 2], eval[3]),
        ([2, 0, 2, 2], eval[3]),
        ([0, 2, 2, 2], eval[3]),
        ([1, 1, 1, 0], eval[4]),
        ([1, 1, 0, 1], eval[4]),
        ([1, 0, 1, 1], eval[4]),
        ([0, 1, 1, 1], eval[4]),
        ([1, 1, 0, 0], eval[5]),
        ([1, 0, 1, 0], eval[5]),
        ([1, 0, 0, 1], eval[5]),
        ([0, 1, 1, 0], eval[5]),
        ([0, 1, 0, 1], eval[5]),
        ([0, 0, 1, 1], eval[5]),
        ([2, 2, 0, 0], eval[6]),
        ([2, 0, 2, 0], eval[6]),
        ([2, 0, 0, 2], eval[6]),
        ([0, 2, 2, 0], eval[6]),
        ([0, 2, 0, 2], eval[6]),
        ([0, 0, 2, 2], eval[6]),
    ]
    return p
                 
def evaluate(grid, pattern):
    # Takes a grid and an evaluation pattern
    # Returns the score of the grid following the given pattern
    
    score = 0
    patterns = pattern
    for ligne in grid:
        for i in range(len(ligne) - 3):
            for pattern, pattern_score in patterns:
                if ligne[i:i+4] == pattern:
                    score += pattern_score

    for j in range(len(grid[0])):
        for i in range(len(grid) - 3):
            col = [grid[i+k][j] for k in range(4)]
            for pattern, pattern_score in patterns:
                if col == pattern:
                    score += pattern_score

    for i in range(len(grid) - 3):
        for j in range(len(grid[0]) - 3):
            diag = [grid[i+k][j+k] for k in range(4)]
            for pattern, pattern_score in patterns:
                if diag == pattern:
                    score += pattern_score

    for i in range(len(grid) - 3):
        for j in range(3, len(grid[0])):
            diag = [grid[i+k][j-k] for k in range(4)]
            for pattern, pattern_score in patterns:
                if diag == pattern:
                    score += pattern_score

    return score  

def possibleMoves(grid):
    # Takes a grid
    # Returns an array of all the possible moves that can be done on the given grid (respecting gravity, no tile overlapping..)
    # A move is an array of 2 integers representing the coordinates of the empty tile to fill
    
    possibleMovesArray = []
    bottomRow = grid[5]
    for tile in range (len(bottomRow)):
        if bottomRow[tile] == 0:
            possibleMovesArray.append([5, tile])
    
    for row in range (len(grid) - 1):
        for tile in range (len(grid[row])):
            if grid[row][tile] == 0 and grid[row + 1][tile] != 0:
                possibleMovesArray.append([row, tile])
                
    return possibleMovesArray  

def get_legal_actions(grid):
    # Takes a grid
    # Returns an array of all the possible moves that can be done on the given grid (respecting gravity, no tile overlapping..)
    # A move is an array of 2 integers representing the coordinates of the empty tile to fill
    # Slightly different function to be used in MCTS. We first look at which player has to play next
    
        count1 = 0
        count2 = 0
        for i in grid:
            for j in i:
                if j == 1:
                    count1 += 1
                if j == 2:
                    count2 += 1
        
        if count1 == count2:
            move = 1
        else: 
            move = 2
            
        possibleMovesArray = []
        bottomRow = grid[5]
        for tile in range (len(bottomRow)):
            if bottomRow[tile] == 0:
                possibleMovesArray.append([5, tile, move])
        
        for row in range (len(grid)-1):
            for tile in range (len(grid[row])):
                if grid[row][tile] == 0 and grid[row + 1][tile] != 0:
                    possibleMovesArray.append([row, tile, move])
                    
        return possibleMovesArray       

def random(grid):
    # Takes a grid and returns a move randomly chosen among all the possible moves
    action = rd.choice(possibleMoves(grid))
    return action

def display_grid(grid):
    # Takes a grid
    # Creates a pretty display of the grid using Pygame
    
    EMPTY_COLOR = (255, 255, 255)  # White
    PLAYER1_COLOR = (0, 199, 140)  # Turquoise
    PLAYER2_COLOR = (235, 64, 128) # Pink
    
    pygame.init()
    pygame.display.set_caption("Connect 4")
    window_size = (len(grid[0]) * 100, len(grid[1]) * 100 - 100)
    window = pygame.display.set_mode(window_size)
    window.fill((0, 0, 0))  # Black

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            tile_value = grid[row][col]
            tile_color = EMPTY_COLOR

            if tile_value == 1:
                tile_color = PLAYER1_COLOR
            elif tile_value == 2:
                tile_color = PLAYER2_COLOR

            pygame.draw.circle(window, tile_color, (col * 100 + 100 // 2, row * 100 + 100 // 2), 100 // 2 - 5)

    pygame.display.flip()

#################################################################

# MINIMAX

#################################################################

def miniMax(root, maxDepth, pattern):
    # MiniMax algorithm, for more information following this link: https://en.wikipedia.org/wiki/Minimax
    evaluation, move = playerMax(root, maxDepth, pattern)
    return move

def playerMax(n, p, pattern):
    possibleMovesArray = possibleMoves(n)
    if possibleMovesArray == [] or p == 0:
        return evaluate(n, pattern), "null"

    u = float('-inf')
    bestMove = "null"
    
    for move in possibleMovesArray:
        newgrid = makeMove(move, n, 1)
        eval, useless = playerMin(newgrid, p-1, pattern)
        
        if eval > u:
            u = eval
            bestMove = move
            
    return u, bestMove

def playerMin(n, p, pattern):
    possibleMovesArray = possibleMoves(n)
    if possibleMovesArray == [] or p == 0:
        return evaluate(n, pattern), "null"   
    
    u = float('inf')
    bestMove = "null"
    
    for move in possibleMovesArray: 
        newgrid = makeMove(move, n, 2)
        eval, useless = playerMax(newgrid, p-1, pattern)
        
        if eval < u:
            u = eval
            bestMove = move
    return u, bestMove

#################################################################

# ALPHA BETA

#################################################################

def alphaBeta(root, maxDepth, pattern):
    # AlphaBeta algorithm, for more information please follow this link : https://en.wikipedia.org/wiki/Alphabeta
    eval, move = playerMaxAB(root, maxDepth, float('-inf'), float('inf'), pattern)    
    return move

def playerMaxAB(n, p, alpha, beta, pattern):
    possibleMovesArray = possibleMoves(n)
    if possibleMovesArray == [] or p == 0:
        return evaluate(n, pattern), "null"
    
    u = float('-inf')
    bestMove = "null"
    
    for move in possibleMovesArray:
        newgrid = makeMove(move, n, 1)
        eval, useless = playerMinAB(newgrid, p-1, alpha, beta, pattern)

        if eval > u:
            u = eval
            bestMove = move
            
        if u >= beta:
            return u, bestMove
        
        alpha = max(alpha, u)
    
    return u, bestMove
  
def playerMinAB(n, p, alpha, beta, pattern):
    possibleMovesArray = possibleMoves(n)
    if possibleMovesArray == [] or p == 0:
        return evaluate(n, pattern), "null"
    
    u = float('inf')
    bestMove = "null"
    
    for move in possibleMovesArray:
        newgrid = makeMove(move, n, 2)
        eval, useless = playerMaxAB(newgrid, p-1, alpha, beta, pattern)

        if eval < u:
            u = eval
            bestMove = move
            
        if u <= alpha:
            return u, bestMove
        
        beta = min(beta, u)
    
    return u, bestMove
       
#################################################################

# MONTE CARLO TREE SEARCH - UCT

#################################################################
        
def mainMCTS(initial_state):
    # MCTS initialization
    root = MonteCarloUCT(state = initial_state)
    selected_node = root.best_action()
    return selected_node.state

class MonteCarloUCT():
    def __init__(self, state, parent=None, parent_action=None):
        # Initialisation
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        # Getting all the legal moves
        self._untried_actions = get_legal_actions(self.state)
        return self._untried_actions
    
    def q(self):
        # Q calculus
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        # N recovery
        return self._number_of_visits
    
    def expand(self):
        # MCTS expanding
        action = self._untried_actions.pop()
        next_state = move(self.state, action)
        child_node = MonteCarloUCT(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 
    
    def is_terminal_node(self):
        # is the node terminal
        return is_game_over(self.state)
    
    def rollout(self):
        # MCTS rollout stage
        current_rollout_state = self.state
    
        while not is_game_over(current_rollout_state):
            possible_moves = get_legal_actions(current_rollout_state)
            action = self.rollout_policy(possible_moves)
            current_rollout_state = move(current_rollout_state, action)
        return game_result(current_rollout_state)
    
    def backpropagate(self, result):
        # MCTS backpropagation stage
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
            
    def is_fully_expanded(self):
        # Is it fully expanded 
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=0.1):
        # weights computing
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        # Seeking the best move
        simulation_no = 35000
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)
               
def play_gameMCTS():
    # Plays MCTS against itself
    # Displays the grid every turn
    state = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]
    
    while(not is_game_over(state)):
        state = mainMCTS(state)
        showGrid(state)
        print('      ')

#################################################################

# Functions used for simulation purposes

#################################################################

def MCTSvsMCTS():
    # IA MCTS playing against IA MCTS
    running = True
    grid = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]
    
    while running:

        display_grid(grid)

        if is_game_over(grid):
            running = False    
            print('Game Over')
            pygame.time.wait(2000)
            continue
        grid = mainMCTS(grid)

    pygame.quit()
    
def AIvsAI(type1, maxP1, eval1, type2, maxP2, eval2):
    # Plays a game between 2 AI : first one of type type1, with a max depth maxP1 and following an evaluation function eval1
    # Against an AI of type type2, with a max depth maxP2 and following an evaluation function eval2
    # If the type is MCTS, the maxP and eval don't matter as they won't be taken into account
    # Type choices : "minimax", "alphabeta", "random", "mcts"
    
    grid = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]
    
    pattern1 = convertEvalIntoPattern(eval1)
    pattern2 = convertEvalIntoPattern(eval2)
    isItOver = game_over(grid)
    
    while(not isItOver[0]):
    
        if (type1 == "minimax"):
            action = miniMax(grid, maxP1, pattern1)
            newgrid = makeMove(action, grid, 1)
        elif (type1 == "alphabeta"):
            action = alphaBeta(grid, maxP1, pattern1)
            newgrid = makeMove(action, grid, 1)
        elif (type1 == "random"):
            action = random(grid)
            newgrid = makeMove(action, grid, 1)
        elif (type1 == "mcts"):
            newgrid = mainMCTS(grid)
            

        if (type2 == "minimax"):
            action = miniMax(newgrid, maxP2, pattern2)
            grid = makeMove(action, newgrid, 2)
        elif (type2 == "alphabeta"):
            action = alphaBeta(newgrid, maxP2, pattern2)
            grid = makeMove(action, newgrid, 2)
        elif (type2 =="random"):
            action = random(newgrid)
            grid = makeMove(action, newgrid, 2)
        elif (type2 == "mcts"):
            grid = mainMCTS(newgrid)
            
        isItOver = game_over(grid)   
        
    return isItOver[1]

def AIvsAIwithDisplay(type1, maxP1, eval1, type2, maxP2, eval2):
    # Same function as the one above but adding a Pygame display
    
    running = True
    grid = [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]

    pattern1 = convertEvalIntoPattern(eval1)
    pattern2 = convertEvalIntoPattern(eval2)
    while running:

        display_grid(grid)

        if is_game_over(grid):
            running = False    
            print('Game Over')
            pygame.time.wait(2000)
            continue
        
        if (type1 == "minimax"):
            action = miniMax(grid, maxP1, pattern1)
            newgrid = makeMove(action, grid, 1)
        elif (type1 == "alphabeta"):
            action = miniMax(grid, maxP1, pattern1)
            newgrid = makeMove(action, grid, 1)
        elif (type1 == "random"):
            action = random(grid)
            newgrid = makeMove(action, grid, 1)
        elif (type1 == "mcts"):
            newgrid = mainMCTS(grid)
        
        display_grid(newgrid)   

        if (type2 == "minimax"):
            action = miniMax(newgrid, maxP2, pattern2)
            grid = makeMove(action, newgrid, 2)
        elif (type2 == "alphabeta"):
            action = miniMax(newgrid, maxP2, pattern2)
            grid = makeMove(action, newgrid, 2)
        elif (type2 =="random"):
            action = random(newgrid)
            grid = makeMove(action, newgrid, 2)
        elif (type2 == "mcts"):
            grid = mainMCTS(newgrid)
            
    pygame.quit()        
    
#################################################################

# Functions used for research purposes and to play with

#################################################################

def AIvsRandom(ai, depth, games):
    # Takes an AI type (minimax, alphabeta, mcts, random), a depth for the concerned AIs and an amount of games
    # Simulates X games between the give AI and a player making random moves
    # Returns the amount of victories and ties (defeats can then be calculated like so : defeats = games - (wins + ties))
    
    eval = [1000,    # win reward
        -1000,       # lose reward
        200,         # blocking opponent reward    
        -100,        # opponent makes a clutch move reward
        100,         # move considered clutch reward
        10,          # move considered good reward
        -10          # opponent makes a good move reward
        ]

    wins = 0
    ties = 0
    
    for i in range(games):
        print("Computing.. " + str(i/games*100) + " done")
        game = AIvsAI(ai, depth, eval, 'random', 0, eval)
        if game == 1:
            wins += 1
        elif game == "it's a tie":
            ties += 1

    return wins, ties

def DepthTesting(minDepth, maxDepth):
    # Simulates all possible AlphaBeta matches within given depth limits (minDepth and maxDepth included)
    
    res = []
    eval = [1000,    # win reward
        -1000,       # lose reward
        200,         # blocking opponent reward    
        -100,        # opponent makes a clutch move reward
        100,         # move considered clutch reward
        10,          # move considered good reward
        -10          # opponent makes a good move reward
        ]

    for i in range(minDepth, maxDepth+1):
        for j in range(i, maxDepth+1):
            print("Ongoing match " + str(i) + " vs " + str(j))
            gameRes = [i, j, 0, 0]
            homeLeg = AIvsAI("alphabeta", i, eval, "alphabeta", j, eval)
            awayLeg = AIvsAI("alphabeta", j, eval, "alphabeta", i, eval)
            gameRes[2] = homeLeg
            gameRes[3] = awayLeg
            res.append(gameRes)
    return(res)     

evalTheKing = [1000,    # win reward
        -1000,          # lose reward
        123.19,         # blocking opponent reward
        -110.25,        # opponent makes a clutch move reward
        138.57,         # move considered clutch reward
        44.1,           # move considered good reward
        -21             # opponent makes a good move reward
        ] 

evalChallenger = [1000,    # win reward
        -1000,             # lose reward
        123.19,            # blocking opponent reward
        -110.25,           # opponent makes a clutch move reward
        138.57,            # move considered clutch reward
        44.1,              # move considered good reward
        -21                # opponent makes a good move reward
        ] 


# findingPerfection(evalTheKing, evalChallenger, evalChallenger)

def findingPerfection(evalKing, evalChall, evalPreviousChall):
    # Discrete optimization function that seeks for the best alphabeta/minimax evaluation 
    # This function simulates games between evalTheKing (best evaluation ever found) against evalChallenger,
    # If evalChallenger wins, then it becomes the new evalTheKing
    # Otherwise, we take a random evaluation value and we add +-5% to its initial value, and we repeat
    # The evaluation evalTheKing has been found after roughly 40h of computing
    
    pattern1 = evalKing
    pattern2 = evalChall

    homeLeg = AIvsAI("alphabeta", 2, pattern1, "alphabeta", 5, pattern2)
    awayLeg = AIvsAI("alphabeta", 2, pattern2, "alphabeta", 5, pattern1)
        
    if homeLeg == 2 and awayLeg == 1:
        print(evalChall)
        winsound.Beep(440, 1000) # It can take a huge amount of time to find a better evaluation, so if one is found, it makes a cool BEEEP
        return evalChall
    if homeLeg == 1 and awayLeg == 2:
        findingPerfection(evalKing, evalPreviousChall, evalPreviousChall)
    else:
        evalPreviousChall = evalChall
        randomVariable = rd.randint(2, 6)
        plusOrMinus = rd.choice([-1, 1])
        valueWereLookingFor = evalChall[randomVariable]
        replacedValue = valueWereLookingFor * 5 / 100
        evalChall[randomVariable] = valueWereLookingFor + plusOrMinus * replacedValue
        findingPerfection(evalKing, evalChall, evalPreviousChall)   
        

#%%