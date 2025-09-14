class tree_structure:
    def __init__(self, board, player, move, depth):
        self.board = board
        self.player = player
        self.move = move
        self.depth = depth
        self.children = []

    def children(self):
        #all the childeren the nodes can have/ all possible moves a player have for now
        #childeren nodes having childeren to complete the tree/ each child has a new board, new tree_structure and new childeren
        pass

