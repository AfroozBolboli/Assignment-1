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
            print(self.children)
            #print('child', move+1, new_board)


class MinMaxPlayer(PlayerController):
    """Class for the MinMax player using the MinMax algorithm
    Inherits from PlayerController
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
        self.depth = depth

    def make_move(self, board: Board) -> int:
        """
        Decides the best move by running the MinMax algorithm to the given depth.

        Args:
            board (Board): current board state

        Returns:
            int: column to play in
        """
        # root of the game tree
        root = GameTreeNode(board)

        best_value = -np.inf
        best_move = 0

        # all possible moves for the current player maybe i am redoing cause we have it in 
        #the tree as well
        for move in board.get_possible_moves():
            new_board = board.get_new_board(move, self.player_id)
            child_node = GameTreeNode(new_board, move, self.player_id)
            value = self._minmax(child_node, self.depth - 1, False)  # False = minimizing opponent

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def _minmax(self, node: GameTreeNode, depth: int, maximizing_player: bool) -> int:
        
        #recurssion demon MinMax function.
        
        winner = Heuristic.winning(node.board.get_board_state(), self.game_n)
        
        # depth 0
        if depth == 0 or winner != 0:
            return self.heuristic.evaluate_board(self.player_id, node.board)

        if maximizing_player:
            max_eval = -np.inf
            for move in node.board.get_possible_moves():
                new_board = node.board.get_new_board(move, self.player_id)
                child = GameTreeNode(new_board, move, self.player_id)
                eval = self._minmax(child, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = np.inf
            opponent = 1 if self.player_id == 2 else 2
            for move in node.board.get_possible_moves():
                new_board = node.board.get_new_board(move, opponent)
                child = GameTreeNode(new_board, move, opponent)
                eval = self._minmax(child, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
    

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

    def make_move(self, board: 'Board') -> int:
        best_val, best_move = -np.inf, None
        alpha, beta = -np.inf, np.inf

        possible_moves = board.get_possible_moves()
        if not possible_moves:
            return 0  # fallback if no valid moves
        best_move = possible_moves[0]  # ensure a valid move

        for move in possible_moves:
            new_board = board.get_new_board(move, self.player_id)
            val = self._alphabeta(new_board, self.depth - 1, alpha, beta, False)
            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)

        return best_move

    def _alphabeta(self, board: 'Board', depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        # stop at depth or if game is won
        from app import winning  # import the winning function

        winner = winning(board.get_board_state(), self.game_n)
        if depth == 0 or winner != 0:
            if winner == self.player_id:
                return 1e6  # positive
            elif winner == 3 - self.player_id:
                return -1e6  #negative
            else:
                return 0  # depth limit reached

        possible_moves = board.get_possible_moves()
        if not possible_moves:
            return 0  # no moves left

        if maximizing:
            value = -np.inf
            for move in possible_moves:
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
            for move in possible_moves:
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
        