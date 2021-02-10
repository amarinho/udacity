"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import isolation


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - ( 2* opp_moves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if opp_moves == 0:
        opp_moves = float("-inf")

    return float(own_moves / (2 * opp_moves))


def center_score(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    center = center_score(game, player)

    result = own_moves - ( 2* opp_moves) - center

    return float(result)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    NO_LEGAL_MOVE = (-1, -1)

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    @staticmethod
    def __is_terminal_state(game: isolation.Board, depth: int) -> bool:
        """
        Checks if game has reached end state
        :param game: An instance of the Isolation game `Board` class representing the current
        game state
        :param depth: Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        :return: True if terminal state
        """
        if not game.get_legal_moves() or depth <= 0:
            return True

        return False

    def __check_time(self) -> None:
        """
        Checks if it reached time threshold
        :return: None
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def __min_value(self, game: isolation.Board, depth: int) -> float:
        """
        Calculates minimal utility value
        :param game: An instance of the Isolation game `Board` class representing the current
        game state
        :param depth: Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        :return: min utility value
        """
        self.__check_time()

        # if TERMINAL-TEST(state) then return UTILITY(state)
        if self.__is_terminal_state(game, depth):
            return self.score(game, self)

        # v ← ∞
        min_val = float("inf")

        # for each a in ACTIONS(state) do
        for move in game.get_legal_moves():
            # v ← MIN(v, MAX-VALUE(RESULT(state, a)))
            min_val = min(min_val, self.__max_value(game.forecast_move(move), depth - 1))

        # return v
        return min_val

    def __max_value(self, game: isolation.Board, depth):
        """
        Calculates minimal utility value
        :param game: An instance of the Isolation game `Board` class representing the current
        game state
        :param depth: Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        :return: min utility value
        """
        self.__check_time()

        # if TERMINAL-TEST(state) then return UTILITY(state)
        if self.__is_terminal_state(game, depth):
            return self.score(game, self)

        # v ← −∞
        max_val = float("-inf")

        # for each a in ACTIONS(state) do
        for move in game.get_legal_moves():
            # v ← MAX(v, MIN-VALUE(RESULT(state, a)))
            max_val = max(max_val, self.__min_value(game.forecast_move(move), depth - 1))

        # return v
        return max_val

    def minimax(self, game: isolation.Board, depth: int) -> (int, int):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return self.NO_LEGAL_MOVE

        # return arg max a ∈ ACTIONS(s) MIN-VALUE(RESULT(state, a))
        actions = [(self.__min_value(game.forecast_move(move), depth - 1), move)
                   for move in legal_moves]

        score, best_move = max(actions)

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    NO_LEGAL_MOVE = (-1, -1)

    def get_move(self, game: isolation.Board, time_left) -> (int, int):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = self.NO_LEGAL_MOVE

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            search_depth = 1

            while (True):
                move = self.alphabeta(game, search_depth)

                if move != (-1, -1):
                    best_move = move

                search_depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    @staticmethod
    def __is_terminal_state(game: isolation.Board, depth: int) -> bool:
        """
        Checks if game has reached end state
        :param game: An instance of the Isolation game `Board` class representing the current
        game state
        :param depth: Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        :return: True if terminal state
        """
        if not game.get_legal_moves() or depth <= 0:
            return True

        return False

    def __check_time(self) -> None:
        """
        Checks if it reached time threshold
        :return: None
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def __max_value(self, game, depth, alpha, beta) -> (float, (int, int)):
        """
        Calculates maximum utility value
        :param game:
        :param depth:
        :param alpha:
        :param beta:
        :return:
        """

        self.__check_time()

        # if TERMINAL-TEST(state) then return UTILITY(state)
        if self.__is_terminal_state(game, depth):
            return self.score(game, self), self.NO_LEGAL_MOVE

        # v ← −∞
        max_val = float("-inf")
        best_move = self.NO_LEGAL_MOVE

        # for each a in ACTIONS(state) do
        for move in game.get_legal_moves():
            # v ← MAX(v, MIN-VALUE(RESULT(state, a), α, β))
            score, _ = self.__min_value(game.forecast_move(move), depth - 1, alpha, beta)

            if score > max_val:
                max_val, best_move = score, move

            # if v ≥ β then return v
            if max_val >= beta:
                return max_val, best_move

            # α ← MAX(α, v)
            alpha = max(alpha, max_val)

        # return v
        return max_val, best_move

    def __min_value(self, game, depth, alpha, beta) -> (float, (int, int)):
        """
        Calculates minimal utility value
        :param game:
        :param depth:
        :param alpha:
        :param beta:
        :return:
        """

        self.__check_time()

        # if TERMINAL-TEST(state) then return UTILITY(state)
        if self.__is_terminal_state(game, depth):
            return self.score(game, self), self.NO_LEGAL_MOVE

        # v ← +∞
        min_val = float("inf")
        best_move = self.NO_LEGAL_MOVE

        # for each a in ACTIONS(state) do
        for move in game.get_legal_moves():

            # v ← MIN(v, MAX-VALUE(RESULT(state, a), α, β))
            score, _ = self.__max_value(game.forecast_move(move), depth - 1, alpha, beta)

            if score < min_val:
                min_val, best_move = score, move

            # if v ≤ α then return v
            if min_val <= alpha:
                return min_val, best_move

            # β ← MIN(β, v)
            beta = min(beta, min_val)

        # return v
        return min_val, best_move

    def alphabeta(self, game: isolation.Board, depth: int, alpha=float("-inf"),
                  beta=float("inf")) -> (int, int):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # v ← MAX-VALUE(state, −∞, +∞)
        score, move = self.__max_value(game, depth, alpha, beta)

        return move
