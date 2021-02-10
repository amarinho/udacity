#!/usr/bin/env python3
__author__ = 'andre.marinho'
__project__ = 'Isolation'

""" Code from the course Artificial Intelligence Nanodegree Program by Udacity.
Isolation gaming player. 

 The board is a rectangle with 6 cells inside it. The bottom right cell starts the game as blocked.
 
 Each cell on the board can be described by an ordered pair (x, y). 
 Thus (0, 0) is the top-left corner; (2, 1) is the bottom right corner (the blocked cell).
 
 Each turn, a player can move like a queen in chess (in any of the eight directions) as long as their path does 
 not cross a cell already filled in (previously occupied) or currently occupied. 
 
"""

xlim, ylim = 3, 2  # board dimensions


class GameState:



    def __init__(self):
        self._board = [[0] * ylim for _ in range(xlim)]
        self._board[-1][-1] = 1  # block lower-right corner
        self._parity = 0
        self._player_locations = [None, None]

    def forecast_move(self, move):
        """ Executes a move and returns a new board object with the specified move
        applied to the current game state.

        :param tuple move: The target position for the active player's next move
        :return: a new board object with the specified move applied to the current game state.
        """
        return self._board

    def get_legal_moves(self):
        """ Return a list of all legal moves available to the
        active player.  Each player should get a list of all
        empty spaces on the board on their first move, and
        otherwise they should get a list of all open spaces
        in a straight line along any row, column or diagonal
        from their current position. (Players CANNOT move
        through obstacles or blocked squares.) Moves should
        be a pair of integers in (column, row) order specifying
        the zero-indexed coordinates on the board.
        :return: a list of all legal moves available to the active player.
        """
        pass