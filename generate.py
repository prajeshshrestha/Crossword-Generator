import copy
import random
import heapq
import time

from collections import deque
from queue import PriorityQueue
from pprint import pprint

from crossword import Crossword, Variable

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword

        # setting up the domains for each of the variables

        self.domains = {
            var: self.get_required_length_answers(var.length)
            for var in self.crossword.variables
        }

    # enforcing the node consistency here
    def get_required_length_answers(self, ans_length):
        output = []
        temp_word_list = list(self.crossword.words.copy())
        random.shuffle(temp_word_list)
        for word in temp_word_list:
            if len(word) == ans_length:
                output.append(word)
        random.shuffle(output)
        output = output[:500]
        print(len(output))
        return output

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("██", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("Roboto-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)


    ### here starts the main solving category

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        # self.enforce_node_consistency() # already being handled during initialization
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.crossword.variables:
            valid_words = []
            for word in self.domains[variable]:
                if len(word) == variable.length:
                    valid_words.append(word)
            self.domains[variable] = valid_words

    def revise(self, x, y):
        """
            Make variable `x` arc consistent with variable `y`.
            To do so, remove values from `self.domains[x]` for which there is no
            possible corresponding value for `y` in `self.domains[y]`.

            Return True if a revision was made to the domain of `x`; return
            False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        y_chars = {word[overlap[1]] for word in self.domains[y]}  # Precompute the y's second character set
        self.domains[x] = {word for word in self.domains[x] if word[overlap[0]] in y_chars}
        if len(self.domains[x]) < len(self.domains[y]):
            revised = True
        return revised

    def ac3(self, arcs=None):
        if arcs is None:
            arcs = deque([(v1, v2) for v1 in self.crossword.variables for v2 in self.crossword.neighbors(v1)])
        else:
            arcs = deque(arcs)

        while arcs:
            x, y = arcs.popleft()  # Efficient pop from the left
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        complete = True
        vars_in_assignment = set(var for var in assignment)
        # Checking if all vars in the crossword has been assigned
        if vars_in_assignment != self.crossword.variables:
            complete = False
        for var in assignment:
            # making sure no var is empty
            assert isinstance(assignment[var], str)
            if not assignment[var]:
                complete = False
        return complete

    def consistent(self, assignment):
        """
          Return True if `assignment` is consistent (i.e., words fit in crossword
          puzzle without conflicting characters); return False otherwise.
        """
        values = set()
        for var, word in assignment.items():
            if word in values or len(word) != var.length:
                return False
            values.add(word)
            for neighbor in self.crossword.neighbors(var):
                overlap = self.crossword.overlaps[var, neighbor]
                if neighbor in assignment and assignment[var][overlap[0]] != assignment[neighbor][overlap[1]]:
                    return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        word_cost = []

        for i, word in enumerate(self.domains[var]):
            word_cost.append([word, 0])
            unassigned_neighbors = list(filter(lambda neighbor: neighbor not in assignment , self.crossword.neighbors(var)))
            for neighbor in unassigned_neighbors:
                for n_word in self.domains[neighbor]:
                    cors = self.crossword.overlaps[(var, neighbor)]
                    if len(word) > cors[0] and len(n_word) > cors[1]:
                        if word[cors[0]] != n_word[cors[1]]:
                            word_cost[i][1] += 1

        # var_words = list(sorted(word_cost, key=lambda word: word_cost[word]))
        sorted_word_cost = sorted(word_cost, key=lambda word: word[1])
        ordered_domain_values = list(map(lambda word: word[0], sorted_word_cost))

        return ordered_domain_values

    def select_unassigned_variable(self, assignment):
        """
        Ordering:
          Return an unassigned variable not already part of `assignment`.
          Choose the variable with the minimum number of remaining values - MRV
          in its domain. If there is a tie, choose the variable with the highest
          degree. If there is a tie, any of the tied variables are acceptable
          return values.
        """
        var_penalty = {}
        for var in self.crossword.variables:
            if var not in assignment:
                var_penalty[var] = len(self.domains[var])
        vars = sorted(var_penalty, key= lambda v: var_penalty[v])
        # if the two first variables have the same domain size
        if len(vars) > 1 and var_penalty[vars[0]] == var_penalty[vars[1]]:
            # Check number of neighbors and return highest degree
            if len(self.crossword.neighbors(vars[0])) < len(self.crossword.neighbors(vars[1])):
                return vars[1]
        return vars[0]

    # Modify the backtrack method
    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment  # base case
        
        self.ac3()
        var = self.select_unassigned_variable(assignment)

        # for value in self.order_domain_values(var, assignment):

        for value in self.domains[var]:
            new_assignment = assignment.copy()  # or dict(assignment)
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        return None


def main(grid, word_list, output):

    # # Check usage
    # if len(sys.argv) not in [3, 4]:
    #     sys.exit("Usage: python generate.py structure words [output]")

    # # Parse command-line arguments
    # structure = sys.argv[1]
    # words = sys.argv[2]
    # output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(grid, word_list)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)

grid = [['', '', '', '-'],
        ['A', 'D', 'E', 'A'],
        ['S', 'A', 'E', 'N'],
        ['O', 'C', 'D', 'S'],
        ['-', '', '', '']]

main(grid, "./all_answers.txt", "./")