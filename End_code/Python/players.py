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


#GAMETREE NODE
#TO ILLUSTRATE HOW TO MAKE A GAMETREE
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
            #print(self.children)
            #print('child', move+1, new_board)


#MINMAX CODE
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
        """Gets the column for the player to play in
        Args:
            board (Board): the current board
        Returns:
            int: column to play in
        """
        possible_moves = board.get_possible_moves()
        if not possible_moves:
            return 0
        
        max_value: float = -np.inf # negative infinity
        max_move: int = 0
        for col in range(board.width):
            if board.is_valid(col):
                new_board: Board = board.get_new_board(col, self.player_id)
                # Calling the recursion depth-1 because we have already evaluated one level 
                # Maximmize false because next round is the other player's turn
                value: float = self._minmaxAlgorithm(new_board, self.depth - 1, is_maximizing=False)
                if value > max_value:
                    max_value = value 
                    max_move = col
        return max_move

    def _minmaxAlgorithm(self, board: Board, depth: int, is_maximizing:bool):
        possible_moves = board.get_possible_moves()
        if not possible_moves:
            return 0
        from app import winning
        winner = winning(board.get_board_state(), self.game_n)
        if depth == 0 or winner != 0:
            if winner == self.player_id:
                return 1e6
            elif winner == (3 - self.player_id):
                return -1e6
            elif winner == -1:
                return 0
            else:
                return self.heuristic.evaluate_board(self.player_id, board)
            
        if is_maximizing:
            max_value = -np.inf
            for col in range(board.width):
                if board.is_valid(col):
                    child = board.get_new_board(col, self.player_id)
                    eval_score = self._minmaxAlgorithm(child, depth - 1, is_maximizing=False)
                    max_value = max(eval_score, max_value)
            return max_value
        else:
            min_value = np.inf
            if self.player_id == 1:
                other_player = 2
            else:
                other_player = 1
            for col in range(board.width):
                if board.is_valid(col):
                    child = board.get_new_board(col, other_player)
                    eval_score = self._minmaxAlgorithm(child, depth - 1, is_maximizing=True)
                    min_value = min(eval_score, min_value)
            return min_value
        

#ALPHABETA PRUNING CODE
#LARGELY SIMILAIR TO MINMAX WITH ADDED BOUNDS        
class AlphaBetaPlayer(PlayerController):
    """Class for the MinMax player using the MinMax algorithm with alpha-beta pruning
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
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        possible_moves = board.get_possible_moves()
        if not possible_moves:
            return 0

        max_value: float = -np.inf  
        max_move: int = 0
        alpha, beta = -np.inf, np.inf

        for col in range(board.width):
            if board.is_valid(col):
                new_board: Board = board.get_new_board(col, self.player_id)
                value: float = self._alphabeta(new_board, self.depth - 1, alpha, beta, is_maximizing=False)
                if value > max_value:
                    max_value = value
                    max_move = col
                alpha = max(alpha, max_value)

        return max_move

    def _alphabeta(self, board: Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        # recursion demon AlphaBeta function
        from app import winning
        winner = winning(board.get_board_state(), self.game_n)

        if depth == 0 or winner != 0:
            if winner == self.player_id:
                return 1e6
            elif winner == 3 - self.player_id:
                return -1e6
            elif winner == -1:
                return 0
            else:
                return self.heuristic.evaluate_board(self.player_id, board)

        if is_maximizing:
            max_value = -np.inf
            for col in range(board.width):
                if board.is_valid(col):
                    child = board.get_new_board(col, self.player_id)
                    eval_score = self._alphabeta(child, depth - 1, alpha, beta, is_maximizing=False)
                    max_value = max(max_value, eval_score)
                    alpha = max(alpha, max_value)
                    if alpha >= beta:
                        break  # beta cutoff
            return max_value
        else:
            min_value = np.inf
            opponent = 3 - self.player_id
            for col in range(board.width):
                if board.is_valid(col):
                    child = board.get_new_board(col, opponent)
                    eval_score = self._alphabeta(child, depth - 1, alpha, beta, is_maximizing=True)
                    min_value = min(min_value, eval_score)
                    beta = min(beta, min_value)
                    if beta <= alpha:
                        break  # alpha cutoff
            return min_value

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
        