from heuristics import Heuristic, SimpleHeuristic, ProgressiveHeuristic
from players import PlayerController, HumanPlayer, MinMaxPlayer, AlphaBetaPlayer
from board import Board
from typing import List
import numpy as np
from numba import jit

def start_game(game_n: int, board: Board, players: List[PlayerController]) -> int:
    """Starting a game and handling the game logic"""
    print('Start game!')

    current_player_index: int = 0
    winner: int = 0

    while winner == 0:
        current_player: PlayerController = players[current_player_index]
        move: int = current_player.make_move(board)

        while not board.play(move, current_player.player_id):
            move = current_player.make_move(board)

        current_player_index = 1 - current_player_index
        winner = winning(board.get_board_state(), game_n)

    print(board)

    if winner < 0:
        print('Game is a draw!')
    else:
        print(f'Player {current_player} won!')

    for p in players:
        print(f'Player {p} evaluated a boardstate {p.get_eval_count()} times!')

    return winner


@jit(nopython=True, cache=True)
def winning(state: np.ndarray, game_n: int) -> int:
    """Determines whether a player has won, and if so, which one"""
    player: int
    counter: int

    # Vertical check
    for col in state:
        counter = 0
        player = -1
        for field in col[::-1]:
            if field == 0:
                break
            elif field == player:
                counter += 1
                if counter >= game_n:
                    return player
            else:
                counter = 1 
                player = field

    # Horizontal check
    for row in state.T:
        counter = 0
        player = -1
        for field in row:
            if field == 0:
                counter = 0
                player = -1
            elif field == player:
                counter += 1
                if counter >= game_n:
                    return player
            else:
                counter = 1
                player = field

    # Ascending diagonal check
    for i, col in enumerate(state[:- game_n + 1]):
        for j, field in enumerate(col[game_n - 1:]):
            if field == 0:
                continue
            player = field
            for x in range(game_n):
                if state[i + x, j + game_n - 1 - x] != player:
                    player = -1
                    break
            if player != -1:
                return player

    # Descending diagonal check
    for i, col in enumerate(state[game_n - 1:]):
        for j, field in enumerate(col[game_n - 1:]):
            if field == 0:
                continue
            player = field
            for x in range(game_n):
                if state[i + game_n - 1 - x, j + game_n - 1 - x] != player:
                    player = -1
                    break
            if player != -1:
                return player

    # Check for a draw
    if np.all(state[:, 0]):
        return -1  # The board is full, game is a draw

    return 0  # Game is not over


def get_players(game_n: int) -> List[PlayerController]:
    """Gets the two players"""
    heuristic1: Heuristic = SimpleHeuristic(game_n)
    heuristic2: Heuristic = ProgressiveHeuristic(game_n)

    #make sure to edit the player ID when changing players
     # human vs alphabeta
    #p1 = HumanPlayer(1, game_n, heuristic1)
    p1 = AlphaBetaPlayer(1, game_n, depth=4, heuristic=heuristic2) #depth = how many moves ahead the player looks
    p2 = MinMaxPlayer(2, game_n, depth=4, heuristic=heuristic2) #depth = how many moves ahead the player looks


    return [p1, p2]


if __name__ == '__main__':
    W = int(input("Enter a number for the width of the grid:"))
    H = int(input("Enter a number for the height of the grid:"))
    N =  int(input("Enter a number for game_N lower than width and height of the grid:"))
    game_n: int = N
    width: int = W
    height: int = H

    assert 1 < game_n <= min(width, height), 'game_n is not possible'

    board: Board = Board(width, height)
    if (W >2 and H > 2) :
        start_game(game_n, board, get_players(game_n))