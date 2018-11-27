
from sample_players import DataPlayer
import math
import random
import copy
from pprint import pprint

#def custom_score_1(game, player):
#    """Calculate the heuristic value of a game state from the point of view
#    of the given player.
#    This should be the best heuristic function for your project submission.
#    Note: this function should be called from within a Player instance as
#    `self.score()` -- you should not need to call this function directly.
#    Parameters
#    ----------
#    game : `isolation.Board`
#        An instance of `isolation.Board` encoding the current state of the
#        game (e.g., player locations and blocked cells).
#    player : object
#        A player instance in the current game (i.e., an object corresponding to
#        one of the player objects `game.__player_1__` or `game.__player_2__`.)
#    Returns
#    -------
#    float
#        The heuristic value of the current game state to the specified player.
#    """
#    if game.is_loser(player):
#        return float('-inf')
#
#    if game.is_winner(player):
#        return float('inf')
#
#    blanks = game.get_blank_spaces()
#    blanks_n = len(blanks)
#    max_blanks = game.width * game.height
#
#    # Beginning game & Middle game.
#    if blanks_n > max_blanks*0.1:
#        return custom_score_2(game, player)
#    # End game.
#    else:
#        my_tree = get_tree(game, player)
#        opp_tree = get_tree(game, game.get_opponent(player))
#        return float(len(my_tree) - len(opp_tree))
#
#def custom_score_2_1(game, player):
#    """Calculate the heuristic value of a game state from the point of view
#    of the given player.
#    Note: this function should be called from within a Player instance as
#    `self.score()` -- you should not need to call this function directly.
#    Parameters
#    ----------
#    game : `isolation.Board`
#        An instance of `isolation.Board` encoding the current state of the
#        game (e.g., player locations and blocked cells).
#    player : object
#        A player instance in the current game (i.e., an object corresponding to
#        one of the player objects `game.__player_1__` or `game.__player_2__`.)
#    Returns
#    -------
#    float
#        The heuristic value of the current game state to the specified player.
#    """
#    if game.is_loser(player):
#        return float('-inf')
#
#    if game.is_winner(player):
#        return float('inf')
#    
#    my_score, opp_score = 0, 0
#    my_moves = game.get_legal_moves(player)
#    opp_moves = game.get_legal_moves(game.get_opponent(player))
#
#    # Look one more ply ahead and count all the possible moves there.
#    for move in my_moves:
#        my_score += len(get_Lmoves(game, move))
#    
#    for move in opp_moves:
#        opp_score += len(get_Lmoves(game, move))
#
#    return float(my_score - opp_score)
#
#def custom_score_3_1(game, player):
#    """Calculate the heuristic value of a game state from the point of view
#    of the given player.
#    Note: this function should be called from within a Player instance as
#    `self.score()` -- you should not need to call this function directly.
#    Parameters
#    ----------
#    game : `isolation.Board`
#        An instance of `isolation.Board` encoding the current state of the
#        game (e.g., player locations and blocked cells).
#    player : object
#        A player instance in the current game (i.e., an object corresponding to
#        one of the player objects `game.__player_1__` or `game.__player_2__`.)
#    Returns
#    -------
#    float
#        The heuristic value of the current game state to the specified player.
#    """
#    if game.is_loser(player):
#        return float('-inf')
#
#    if game.is_winner(player):
#        return float('inf')
#
#    my_moves = game.get_legal_moves(player)
#    opp_moves = game.get_legal_moves(game.get_opponent(player))
#    my_moves_n = len(my_moves)
#    opp_moves_n = len(opp_moves)
#
#    w, h = game.width / 2., game.height / 2.
#    y, x = game.get_player_location(player)
#    # While considering available moves, try to stay towards the middle.
#    return float(my_moves_n - opp_moves_n) - float(math.sqrt((h - y)**2 + (w - x)**2))*0.25




#class CustomPlayer(DataPlayer):
#    """ Implement your own agent to play knight's Isolation
#
#    The get_action() method is the only required method for this project.
#    You can modify the interface for get_action by adding named parameters
#    with default values, but the function MUST remain compatible with the
#    default interface.
#
#    **********************************************************************
#    NOTES:
#    - The test cases will NOT be run on a machine with GPU access, nor be
#      suitable for using any other machine learning techniques.
#
#    - You can pass state forward to your agent on the next turn by assigning
#      any pickleable object to the self.context attribute.
#    **********************************************************************
#    """
#    def get_action(self, state):
#        """ Employ an adversarial search technique to choose an action
#        available in the current state calls self.queue.put(ACTION) at least
#
#        This method must call self.queue.put(ACTION) at least once, and may
#        call it as many times as you want; the caller will be responsible
#        for cutting off the function after the search time limit has expired.
#
#        See RandomPlayer and GreedyPlayer in sample_players for more examples.
#
#        **********************************************************************
#        NOTE: 
#        - The caller is responsible for cutting off search, so calling
#          get_action() from your own code will create an infinite loop!
#          Refer to (and use!) the Isolation.play() function to run games.
#        **********************************************************************
#        """
#        # TODO: Replace the example implementation below with your own search
#        #       method by combining techniques from lecture
#        #
#        # EXAMPLE: choose a random move without any search--this function MUST
#        #          call self.queue.put(ACTION) at least once before time expires
#        #          (the timer is automatically managed for you)
#        import random
#        self.queue.put(random.choice(state.actions()))



#########################################################################################








#########################################################################################


    

###################################################################################





######################################################################################
## Original Algorithm from Base Algo

def score_greedy_orig(state, play_id):
    own_loc = state.locs[play_id]
    own_liberties = state.liberties(own_loc)
    return len(own_liberties)

def score_minimax_orig(state, play_id):
    own_loc = state.locs[play_id]
    opp_loc = state.locs[1 - play_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)


from isolation.isolation import _WIDTH, _HEIGHT

def distance(state):
    """ minimum distance to the walls """
    own_loc = state.locs[state.ply_count % 2]
    x_player, y_player = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)

    return min(x_player, _WIDTH + 1 - x_player, y_player, _HEIGHT - 1 - y_player)

def custom_score_1(state, play_id):
    own_loc = state.locs[play_id]
    opp_loc = state.locs[1 - play_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)

    dis = distance(state)
    if dis >= 2:
        return 2*len(own_liberties) - len(opp_liberties)
    else:
        # states away from walls from be encouraged, so the weight is bigger
        return len(own_liberties) - len(opp_liberties)





def custom_score_2(state, play_id):
    own_loc = state.locs[play_id]
    opp_loc = state.locs[1 - play_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    # Penalize/reward move count if some moves are in the corner
    return len(own_liberties) - 2 * len(opp_liberties)


#def custom_score_2(state, play_id):
#    blank_spaces = len(state.get_blank_spaces())
#    my_moves = len(state.get_legal_moves(play_id))
#    return float(blank_spaces / 2 + 2 * my_moves)
#
#def custom_score_3(state, play_id):
#    return float(custom_score(state, play_id)) + (float(custom_score_2(state, play_id))/13.0)


######################################################################################

## Random Player

class RandomPlayer(DataPlayer):
    def get_action(self, state):
        """ Randomly select a move from the available legal moves.

        Parameters
        ----------
        state : `isolation.Isolation`
            An instance of `isolation.Isolation` encoding the current state of the
            game (e.g., player locations and blocked cells)
        """
        self.queue.put(random.choice(state.actions()))


## Greedy Player

class GreedyPlayer(DataPlayer):
    """ Player that chooses next move to maximize heuristic score. This is
    equivalent to a minimax search agent with a search depth of one.
    """
    
    def get_action(self, state):
        """Select the move from the available legal moves with the highest
        heuristic score.

        Parameters
        ----------
        state : `isolation.Isolation`
            An instance of `isolation.Isolation` encoding the current state of the
            game (e.g., player locations and blocked cells)
        """
        print(state)
        print(state.liberties(state.locs[self.player_id]))
        self.queue.put(max(state.actions(), key=lambda x: score(state.result(x), self.player_id) ))


class MinimaxPlayer(DataPlayer):
    """ Implement an agent using any combination of techniques discussed
    in lecture (or that you find online on your own) that can beat
    sample_players.GreedyPlayer in >80% of "fair" matches (see tournament.py
    or readme for definition of fair matches).

    Implementing get_action() is the only required method, but you can add any
    other methods you want to perform minimax/alpha-beta/monte-carlo tree search,
    etc.

    **********************************************************************
    NOTE: The test cases will NOT be run on a machine with GPU access, or
          be suitable for using any other machine learning techniques.
    **********************************************************************
    """
    
    def get_action(self, state):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.minimax(state, depth=3))

    def minimax(self, state, depth):
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))


## Alpha Beta

class AlphabetaPlayer(DataPlayer):
    
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.ply_count < 4:
            self.queue.put(random.choice(state.actions()))
        else:
            ###### iterative deepening ######
            depth_limit = 4
            for depth in range(1, depth_limit + 1):
                best_move = self.alpha_beta_search(state, depth)
            self.queue.put(best_move)

            #### no iterative deepening ####
            # self.queue.put(self.alpha_beta_search(state, depth=3))


    def alpha_beta_search(self, state,depth=3):
        """ Return the move along a branch of the game tree that
        has the best possible value.
        """
        def min_value(self, state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return score(state, self.player_id)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(self, state.result(action), alpha, beta, depth-1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value
    
        def max_value(self, state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0: return score(state, self.player_id)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(self, state.result(action), alpha, beta, depth-1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value
    
    
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            value = min_value(self, state.result(action), alpha, beta, depth-1)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move


## Monte Carlo Tree Search

FACTOR = 1.0
iter_limit = 100

class MCTS_Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent

    def add_child(self, child_state, action):
        child = MCTS_Node(child_state, self)
        self.children.append(child)
        self.children_actions.append(action)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_explored(self):
        return len(self.children_actions) == len(self.state.actions())

class MonteCarloPlayer(DataPlayer):
    """
    Implement an agent to play knight's Isolation with Monte Carlo Tree Search
    """
    
    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))
    
    
    def mcts(self, state):
        root = MCTS_Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(iter_limit):
            child = self.tree_policy(root)
            if not child:
                continue
            reward = self.default_policy(child.state)
            self.backup(child, reward)

        idx = root.children.index(self.best_child(root))
        return root.children_actions[idx]
    
    def tree_policy(self, node):
        """
        Select a leaf node.
        If not fully explored, return an unexplored child node.
        Otherwise, return the child node with the best score.
    
        :param node:
        :return: node
        """
        while not node.state.terminal_test():
            if not node.fully_explored():
                return self.expand(node)
            node = self.best_child(node)
        return node
    
    def default_policy(self, state):
        """
        Randomly search the descendant of the state, and return the reward
    
        :param state:
        :return: int
        """
        init_state = copy.deepcopy(state)
        while not state.terminal_test():
            action = random.choice(state.actions())
            state = state.result(action)
        # let the reward be 1 for the winner, -1 for the loser
        # if the init_state.player() wins, it means the action that leads to
        # init_state should be discouraged, so reward = -1.
        return -1 if state._has_liberties(init_state.player()) else 1
    
    def expand(self, node):
        tried_actions = node.children_actions
        legal_actions = node.state.actions()
        for action in legal_actions:
            if action not in tried_actions:
                new_state = node.state.result(action)
                node.add_child(new_state, action)
                return node.children[-1]
    
    def best_child(self, node):
        """
        Find the child node with the best score.
        
        :param node:
        :return: node;
        """
        best_score = float("-inf")
        best_children = []
        for child in node.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            score = exploit + FACTOR * explore
            if score == best_score:
                best_children.append(child)
            elif score > best_score:
                best_children = [child]
                best_score = score
        # if len(best_children) == 0:
        #     print("WARNING - RuiZheng, there is no best child")
        #     return None
        return random.choice(best_children)
    
    def backup(self, node, reward):
        """
        Backpropagation
        Use the result to update information in the nodes on the path.
    
        :param node:
        :param reward: int
        :return:
        """
        while node != None:
            node.update(reward)
            node = node.parent
            reward *= -1



######################################################################################
######################################################################################

CustomPlayer = MonteCarloPlayer
score = custom_score_1
print('Player used : ', CustomPlayer.__name__)
## Originall Algo: RandomPlayer, GreedyPlayer, MinimaxPlayer, AlphabetaPlayer, MonteCarloPlayer
## Score Related Function: score_greedy_orig, score_minimax_orig, custom_score_1, custom_score_2