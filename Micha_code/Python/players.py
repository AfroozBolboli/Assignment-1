from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass

 # PSEUDO CODE FOR GAME TREE
 #class Gametree:
 #board_state
 #move
 #player
 #children: []

 #Function name(self, next_player) --> generate all possible moves for next_player on this board 
 #define possible moves (???)
 #for move in possible moves:
    # new_board = .....
    #apply move to new_board
    # child_node = Gametree(new_board, move, next_player)
    # self.children.append(child_node)    
#return self.children

class GameTreeNode:
    def __init__(self, board, move=None, player=None):
        self.board = board            # Board state at this node
        self.move = move              # Move that led to this node
        self.player = player          # Player who made the move
        self.children = []            # List of child GameTreeNode objects

    def create_children(self, next_player):
        # Generate all possible moves from this board state
        possible_moves = self.board.get_possible_moves()
        for move in possible_moves:
            new_board = self.board.get_new_board(move, next_player)
            child_node = GameTreeNode(new_board, move, next_player)
            self.children.append(child_node)
            #print('child', move+1, new_board)


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        # TODO: implement minmax algortihm!
        # INT: use the functions on the 'board' object to produce a new board given a specific move
        # HINT: use the functions on the 'heuristic' object to produce evaluations for the different board states!
        
        # Example:
        max_value: float = -np.inf # negative infinity
        max_move: int = 0
        for col in range(board.width):
            if board.is_valid(col):
                new_board: Board = board.get_new_board(col, self.player_id)
                value: int = self.heuristic.evaluate_board(self.player_id, new_board)
                if value > max_value:
                    max_move = col

        # This returns the same as
        self.heuristic.get_best_action(self.player_id, board) # Very useful helper function!

        # This is obviously not enough (this is depth 1)
        # Your assignment is to create a data structure (tree) to store the gameboards such that you can evaluate a higher depths.
        # Then, use the minmax algorithm to search through this tree to find the best move/action to take!

        return max_move
    

#PSEUDO CODE FOR ALPHA BETA PRUNING
#Alpha is the best value that the maximizer currently can guarantee at that level 
#Beta is the best value that the minimizer currently can guarantee at that level 
# function alphaBeta(node, depth, α, β, maximizingPlayer):
    #if depth == 0 or node is terminal:
       # return evaluate(node)

   # if maximizingPlayer:
       # d := -∞
       # for each child of node:
            #d := max(d, alphaBeta(child, depth-1, α, β, false))
           #alpha := max(alpha, d)
            #if alpha ≥ β:
                #break   // β cutoff
       # return value
   # else:
       # value := +∞
       # for each child of node:
         #   c := min(c, alphaBeta(child, depth-1, α, β, true))
          #  beta := min(beta, value)
           # if beta ≤ α:
             #   break alpha cutoff
       # return c
    
    

class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from PlayerController
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth

    def make_move(self, board: Board) -> int:
        best_val, best_move = -np.inf, None
        alpha, beta = -np.inf, np.inf

        for move in board.get_possible_moves():
            new_board = board.get_new_board(move, self.player_id)
            val = self._alphabeta(new_board, self.depth - 1, alpha, beta, False)
            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)

        return best_move

    def _alphabeta(self, board: Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        # stop at depth or if no moves left
        if depth == 0 or not board.get_possible_moves():
            return self.heuristic.evaluate_board(self.player_id, board)

        if maximizing:
            value = -np.inf
            for move in board.get_possible_moves():
                value = max(value, self._alphabeta(
                    board.get_new_board(move, self.player_id),
                    depth - 1, alpha, beta, False
                ))
                if value >= beta:  # beta cutoff
                    break
                alpha = max(alpha, value)
            return value
        else:
            opponent = 3 - self.player_id
            value = np.inf
            for move in board.get_possible_moves():
                value = min(value, self._alphabeta(
                    board.get_new_board(move, opponent),
                    depth - 1, alpha, beta, True
                ))
                if value <= alpha:  # alpha cutoff
                    break
                beta = min(beta, value)
            return value



class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)


        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')


        #tree.print_tree()
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        