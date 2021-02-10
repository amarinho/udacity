#!/usr/bin/env python3
__author__ = 'andre.marinho'
__project__ = 'sudoku'

from collections import Counter

""" Code from the course Artificial Intelligence Nanodegree Program by Udacity.
Soduko solving algorithm based on the work of Peter Norvig <https://github.com/norvig>.

A Sudoku puzzle is a grid of 81 squares; the majority of enthusiasts label 
the columns 1-9, the rows A-I, and call a collection of nine squares 
(column, row, or box) a unit, and the squares that share a unit the peers. 
"""

assignments = []


def assign_value(values, box_name, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    :param dict values: a dictionary of the form {'box_name': '123456789', ...}
    :param str box_name: key for receiving the value
    :param str value: value to be assigned
    :return: None.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box_name] == value:
        return values

    values[box_name] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def cross(rows, columns):
    """Returns a list with the permutations of rows x columns.
    :param str rows: string representing rows, i.e 'ABCDEFGHI'.
    :param str columns: string representing columns, i.e '123456789'.
    :return: list with the permutations of rows x columns. For example:
     cross('abc', 'def') will return ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf'].
    """

    # return [row+column for row in rows for column in columns]
    squares = []
    for row in rows:
        for column in columns:
            squares.append(row + column)
    return squares


_rows = 'ABCDEFGHI'
_cols = '123456789'
_boxes = cross(_rows, _cols)

_row_units = [cross(r, _cols) for r in _rows]
# Element example:
# row_units[0] = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
# This is the top most row.

_column_units = [cross(_rows, c) for c in _cols]
# Element example:
# column_units[0] = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1']
# This is the left most column.

_square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
# Element example:
# square_units[0] = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
# This is the top left square.

_left_diagonal = [['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8', 'I9']]
_right_diagonal = [['I1', 'H2', 'G3', 'F4', 'E5', 'D6', 'C7', 'B8', 'A9']]
_unitlist = _row_units + _column_units + _square_units + _left_diagonal + _right_diagonal
_units = dict((s, [u for u in _unitlist if s in u]) for s in _boxes)
_peers = dict((s, set(sum(_units[s], [])) - set([s])) for s in _boxes)


def get_naked_twins(unit, values):
    """Returns a list of naked twins within a unit.
        :param list unit: unit list
        :param dict values: a dictionary of the form {'box_name': '123456789', ...}.

        :return: a list of naked twins boxes, or empty.
    """
    boxes = [box for box in unit if len(values[box]) == 2]

    naked_twins_list = []

    if len(boxes) == 2 and values[boxes[0]] == values[boxes[1]]:
        naked_twins_list = boxes
    elif len(boxes) > 2:
        boxes_dict = dict((k, values[k]) for k in boxes)
        c = Counter(boxes_dict.values())
        for b in boxes:
            val = boxes_dict[b]
            if c[val] == 2:
                naked_twins_list.append(b)

    return naked_twins_list


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    :param dict values: a dictionary of the form {'box_name': '123456789', ...}.

    :return: the values dictionary with the naked twins eliminated from peers.
    """

    new_dict = values
    # Find all instances of naked twins
    for unit in _unitlist:
        # Only boxes with two digits
        boxes = get_naked_twins(unit, values)
        if boxes:
            # Eliminate the naked twins as possibilities for their peers
            for p in unit:
                if new_dict[p] != new_dict[boxes[0]]:
                    d1, d2 = new_dict[boxes[0]][0], new_dict[boxes[0]][1]
                    assign_value(new_dict, p, new_dict[p].replace(d1, ''))
                    assign_value(new_dict, p, new_dict[p].replace(d2, ''))

    return new_dict


def grid_values(grid):
    """Convert grid string into {<box>: <value>} dict with '.' value for empties.

    :param str grid: Sudoku grid in string form, 81 characters long.
    :return: Sudoku grid in dictionary form:
        - keys: Box labels, e.g. 'A1'
        - values: Value in corresponding box, e.g. '8', or '123456789' if it is empty.
    """
    assert len(grid) == 81, 'Input grid must be a string of length 81 (9x9)'

    # dict(zip(boxes, grid))
    dict_grid = {}
    for k, v in zip(_boxes, grid):
        if v == '.':
            dict_grid[k] = '123456789'
        else:
            dict_grid[k] = v

    return dict_grid


def display(values):
    """
    Display the values as a 2-D grid.

    :param: list values: The sudoku in dictionary form.
    :return: None.
    """
    width = 1+max(len(values[s]) for s in _boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in _rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in _cols))
        if r in 'CF': print(line)
    return


def eliminate(values):
    """Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    :param: dict values: Sudoku dictionary of the form {'box_name': '123456789', ...}.
    :return: Resulting Sudoku dictionary of the form {'box_name': '123456789', ...}, after eliminating values.
    """
    new_dict = values
    for k, v in new_dict.items():
        if len(v) == 1:
            for p in _peers[k]:
                assign_value(new_dict, p, new_dict[p].replace(v, ''))

    return new_dict


def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    :param: dict values: Sudoku dictionary of the form {'box_name': '123456789', ...}.
    :return:  Resulting Sudoku in dictionary form after filling in only choices.
    """

    # for each square
    # iterate each box that has more than one possible value
    #   for each possible value checks the number of occurrences in the square

    new_dict = values
    for square in _square_units:
        square_dict = dict((k, new_dict[k]) for k in square)

        for box in square:
            if len(new_dict) > 1:
                for digit in new_dict[box]:
                    occurrences = sum(digit in x for x in square_dict.values())
                    if occurrences == 1:
                        assign_value(new_dict, box, digit)
    return new_dict


def reduce_puzzle(values):
    """Eliminates possible values accordingly to the constraints.

    :param: dict values: Sudoku dictionary of the form {'box_name': '123456789', ...}.
    :return:  Resulting Sudoku in dictionary form after reducing the possible values.
    """

    stalled = False
    new_dict = values
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in new_dict.keys() if len(new_dict[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        new_dict = eliminate(new_dict)

        # Your code here: Use the Only Choice Strategy
        new_dict = only_choice(new_dict)

        # Naked twins strategy
        new_dict = naked_twins(new_dict)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in new_dict.keys() if len(new_dict[box]) == 1])

        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after

        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in new_dict.keys() if len(new_dict[box]) == 0]):
            return False

    return new_dict


def search(values):
    """Using depth-first search and propagation, create a search tree and solve the sudoku..

    :param dict values: Sudoku dictionary of the form {'box_name': '123456789', ...}.
    :return:  Resulting Sudoku in dictionary form after reducing the possible values.
    """

    new_dict = values
    # First, reduce the puzzle using the previous function
    new_dict = reduce_puzzle(new_dict)

    # If error return false
    if new_dict is False:
        return False

    # checks if the Sudoku was solved
    if all(len(new_dict[b]) == 1 for b in _boxes):
        return new_dict

    # Choose one of the unfilled squares with the fewest possibilities
    size, box = min((len(new_dict[b]), b) for b in _boxes if len(new_dict[b]) > 1)

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False),
    # return that answer!
    for value in new_dict[box]:
        recursive_val = dict(new_dict)
        recursive_val[box] = value
        result = search(recursive_val)
        if result:
            return result


def solve(grid):
    """
    Find the solution to a Sudoku grid.

    :param str grid: a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    :return: The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    grid_dict = grid_values(grid)

    return search(grid_dict)


if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
