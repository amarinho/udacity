#!/usr/bin/env python3
__author__ = 'andre.marinho'
__project__ = 'Isolation'

""" Code from the course Artificial Intelligence Nanodegree Program by Udacity.
Isolation gaming player. 
"""

import unittest

from gamestate import *


class GameStateTest(unittest.TestCase):
    def test_something(self):
        print("Creating empty game board...")
        g = GameState()

        b = g.forecast_move((0, 0))

        print(b)

        print("Getting legal moves for player 1...")
        p1_empty_moves = g.get_legal_moves()
        print("Found {} legal moves.".format(len(p1_empty_moves or [])))

        print("Applying move (0, 0) for player 1...")
        g1 = g.forecast_move((0, 0))

        print("Getting legal moves for player 2...")
        p2_empty_moves = g1.get_legal_moves()
        if (0, 0) in set(p2_empty_moves):
            print("Failed\n  Uh oh! (0, 0) was not blocked properly when " +
                  "player 1 moved there.")
        else:
            print("Everything looks good!")

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
