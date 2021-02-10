from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions() -> list:
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            # Action(Load(c, p, a),
            #   PRECOND: At(c, a) ∧ At(p, a) ∧ Cargo(c) ∧ Plane(p) ∧ Airport(a)
            #   EFFECT: ¬ At(c, a) ∧ In(c, p))
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr("At({}, {})".format(c, a)),
                                       expr("At({}, {})".format(p, a)),]
                        precond_neg = []
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        loads.append(load)
            return loads

        def unload_actions() -> list:
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            # Action(Unload(c, p, a),
            #   PRECOND: In(c, p) ∧ At(p, a) ∧ Cargo(c) ∧ Plane(p) ∧ Airport(a)
            #   EFFECT: At(c, a) ∧ ¬ In(c, p)
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr("In({}, {})".format(c, p)),
                                       expr("At({}, {})".format(p, a)),]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(c, a))]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        unloads.append(unload)

            return unloads

        def fly_actions() -> list:
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            # Action(Fly(p, fr, to),
            #   PRECOND: At(p, fr) ∧ Plane(p) ∧ Airport(fr) ∧ Airport(to)
            #   EFFECT: ¬ At(p, fr) ∧ At(p, to))
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """

        # The actions that are applicable to a state are all those whose preconditions are satisfied.
        # The successor state resulting from an action is generated by adding the positive effect
        # literals and deleting the negative effect literals. (In the first-order case, we must apply
        # the unifier from the preconditions to the effect literals.) Note that a single successor
        # function works for all planning problems—a consequence of using an explicit action
        # representation.

        # copied from example_have_cake.py
        possible_actions = []
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
            if is_possible:
                possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action) -> str:
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """

        # copied from example_have_cake.py
        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)
        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        # The goal test checks whether the state satisfies the goal of the planning problem.

        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2
        count = 0
        # Let us consider how to derive relaxed planning problems. Since explicit representations
        # of preconditions and effects are available, the process will work by modifying those representations.
        # (Compare this approach with search problems, where the successor function is
        # a black box.) The simplest idea is to relax the problem by removing all preconditions from
        # the actions. Then every action will always be applicable, and any literal can be achieved in
        # one step (if there is an applicable action—if not, the goal is impossible). This almost implies
        # that the number of steps required to solve a conjunction of goals is the number of unsatisfied
        # goals—almost but not quite, because (1) there may be two actions, each of which deletes
        # the goal literal achieved by the other, and (2) some action may achieve multiple goals. If we
        # combine our relaxed problem with the subgoal independence assumption, both of these issues
        # are assumed away and the resulting heuristic is exactly the number of unsatisfied goals.
        # In many cases, a more accurate heuristic is obtained by considering at least the positive
        # interactions arising from actions that achieve multiple goals. First, we relax the problem further
        # by removing negative effects (see Exercise 11.6). Then, we count the minimum number
        # of actions required such that the union of those actions’ positive effects satisfies the goal.

        # The goal test checks whether the state satisfies the goal of the planning problem.
        kb = PropKB()
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())

        # if it is not goal, increment counter
        not_goal = list(set(self.goal) - set(kb.clauses))
        count = len(not_goal)

        return count


def air_cargo_p1() -> AirCargoProblem:
    # Init(
    #    At(C1, SFO) ¬ At(C1, JFK) ¬ In(C1, P1)  ¬ In(C1, P2)
    # ∧ At(C2, JFK)  ¬ At(C2, SFO) ¬ In(C2, P1)  ¬ In(C2, P2)
    # ∧ At(P1, SFO)  ¬ At(P1, JFK)
    # ∧ At(P2, JFK)  ¬ At(P2, SFO)
    # ∧ Cargo(C1) ∧ Cargo(C2)
    # ∧ Plane(P1) ∧ Plane(P2)
    # ∧ Airport(JFK) ∧ Airport(SFO))
    # Goal(
    #    At(C1, JFK)
    #  ∧ At(C2, SFO))

    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # Init(
    #   At(C1, SFO) ¬ At(C1, JFK) ¬ At(C1, ATL) ¬ In(C1, P1) ¬ In(C1, P2) ¬ In(C1, P3)
    # ∧ At(C2, JFK) ¬ At(C2, SFO) ¬ At(C2, ATL) ¬ In(C2, P1) ¬ In(C2, P2) ¬ In(C2, P3)
    # ∧ At(C3, ATL) ¬ At(C3, JFK) ¬ At(C3, SFO) ¬ In(C3, P1) ¬ In(C3, P2) ¬ In(C3, P3)
    # ∧ At(P1, SFO) ¬ At(P1, JFK) ¬ At(P1, ATL)
    # ∧ At(P2, JFK) ¬ At(P2, SFO) ¬ At(P2, ATL)
    # ∧ At(P3, ATL) ¬ At(P3, JFK) ¬ At(P3, SFO)
    # ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
    # ∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3)
    # ∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL))
    # Goal(
    #    At(C1, JFK)
    # ∧ At(C2, SFO)
    # ∧ At(C3, SFO))

    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),
           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P3, JFK)'),
           expr('At(P3, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # Init(
    #   At(C1, SFO) ¬ At(C1, JFK) ¬ At(C1, ATL) ¬ At(C1, ORD) ¬ In(C1, P1) ¬ In(C1, P2)
    # ∧ At(C2, JFK) ¬ At(C2, SFO) ¬ At(C2, ATL) ¬ At(C2, ORD) ¬ In(C2, P1) ¬ In(C2, P2)
    # ∧ At(C3, ATL) ¬ At(C3, SFO) ¬ At(C3, JFK) ¬ At(C3, ORD) ¬ In(C3, P1) ¬ In(C3, P2)
    # ∧ At(C4, ORD) ¬ At(C4, SFO) ¬ At(C4, JFK) ¬ At(C4, ATL) ¬ In(C4, P1) ¬ In(C4, P2)
    # ∧ At(P1, SFO) ¬ At(P1, JFK) ¬ At(P1, ATL) ¬ At(P1, ORD)
    # ∧ At(P2, JFK) ¬ At(P2, SFO) ¬ At(P2, ATL) ¬ At(P2, ORD)
    # ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
    # ∧ Plane(P1) ∧ Plane(P2)
    # ∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
    # Goal(
    #   At(C1, JFK)
    # ∧ At(C3, JFK)
    # ∧ At(C2, SFO)
    # ∧ At(C4, SFO))

    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(C3, SFO)'),
           expr('At(C3, JFK)'),
           expr('At(C3, ORD)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('At(C4, SFO)'),
           expr('At(C4, JFK)'),
           expr('At(C4, ATL)'),
           expr('In(C4, P1)'),
           expr('In(C4, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


if __name__ == '__main__':

    # copied from example_have_cake.py

    p = [air_cargo_p1(), air_cargo_p2(), air_cargo_p3()]
    print("**** Air Cargo  example problem setup ****")
    print("Initial state for this problem is {}".format(p.initial))
    print("Actions for this domain are:")
    for a in p.actions_list:
        print('   {}{}'.format(a.name, a.args))
    print("Fluents in this problem are:")
    for f in p.state_map:
        print('   {}'.format(f))
    print("Goal requirement for this problem are:")
    for g in p.goal:
        print('   {}'.format(g))
    print()
