# AI-Connect4
Python AIs and learning algorithms (MiniMax, AlphaBeta, MCTS) made for Connect 4.
Each function has a commented explanation for better understanding.

# Wdym AI playing Connect 4 ?

Using evaluation functions, game simulations and by following optimized schemes, we can create and develop AIs to play a board game with the best level possible.
In this project we're specifically focusing on the board game Connect 4, but it's important to understand that this can be adapted to any board game as long as the rules are respected.
Learning how to properly play the game and evaluating the state of a game at any given time might be harder depending on the game you chose.

# MiniMax

If you are curious enough then I advise you to read the following : https://en.wikipedia.org/wiki/Minimax

### TLDR : 
- Create a function that gives a score to a grid wether a victory is likely or not.
- Using this function, simulate the best moves possible from both players until a certain depth that you chose.
- Chose the move that has the best final score.

# AlphaBeta

AlphaBeta has the same logic as MiniMax but computes faster. It is irrelevant to compare the result of games between AlphaBeta and MiniMax. 
However, it's much more interesting to compare the execution of both of these algorithms.

You can also read the following : https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning

### TLDR : Using maths it's possible to highly optimize MiniMax by avoiding useless nodes (we're talking ~3 times faster)

# MCTS

Unlike MiniMax and AlphaBeta that rely on a score related to the state of a game, MCTS randomly simulates games and choses the move giving the highest winrate.

Learn more by reading the following : https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

# Playing with the algorithms

Parameters to play with
- Evaluations for AlphaBeta and MiniMax, any setting can be change to see how the score fluctuates
- Depth for AlphaBeta and MiniMax, execution time and computer level increase exponentially with depth
- Paramter c for MCTS : function best_child(c_parameter), best results with c close to sqrt(2)
- Number of simulations for MCTS : function best_action(), relevant results starting at ~5k, longer execution time starting at ~150k
