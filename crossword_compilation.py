import copy
import random
import heapq
import time
import pandas as pd
import requests
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import puz
import logging
import pickle

from functools import lru_cache
from pprint import pprint
from PIL import Image
from collections import deque, Counter
from queue import PriorityQueue
from pprint import pprint
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

class Variable():

    ACROSS = "across"
    DOWN = "down"

    def __init__(self, i, j, direction, length):
        """Create a new variable with starting point, direction, and length."""
        self.i = i
        self.j = j
        self.direction = direction
        self.length = length
        self.cells = []
        for k in range(self.length):
            self.cells.append(
                (self.i + (k if self.direction == Variable.DOWN else 0),
                 self.j + (k if self.direction == Variable.ACROSS else 0))
            )

    def __hash__(self):
        return hash((self.i, self.j, self.direction, self.length))

    def __eq__(self, other):
        return (
            (self.i == other.i) and
            (self.j == other.j) and
            (self.direction == other.direction) and
            (self.length == other.length)
        )

    def __str__(self):
        return f"({self.i}, {self.j}) {self.direction} : {self.length}"

    def __repr__(self):
        direction = repr(self.direction)
        return f"Variable({self.i}, {self.j}, {direction}, {self.length})"

class Crossword():
    def __init__(self, grid, words_file, file_path = True):
        self.structure = []

        self.height = len(grid) # the number of rows in the grid
        self.width = len(grid[0]) # the number of columns in the grid
        for i in range(len(grid)):
            row = []
            for j in range(len(grid[0])):
                if grid[i][j] == '':
                  row.append(False)
                else:
                  row.append(True)
            self.structure.append(row)

        if not file_path:
            self.words = [word.upper() for word in words_file]

        else:
            # Save vocabulary list
            with open(words_file) as f:
                self.words = set(f.read().upper().splitlines()) # to remove all the duplicates
                self.words = list(self.words)
                for _ in range(5):
                    random.shuffle(self.words)
            self.words = set(self.words)

        # Determine variable set
        self.variables = set()

        for i in range(self.height):
            for j in range(self.width):

                # Vertical words
                starts_word = (
                    self.structure[i][j]
                    and (i == 0 or not self.structure[i - 1][j])
                )
                if starts_word:
                    length = 1
                    for k in range(i + 1, self.height):
                        if self.structure[k][j]:
                            length += 1
                        else:
                            break
                    if length > 1:
                        self.variables.add(Variable(
                            i=i, j=j,
                            direction=Variable.DOWN,
                            length=length
                        ))

                # Horizontal words
                starts_word = (
                    self.structure[i][j]
                    and (j == 0 or not self.structure[i][j - 1])
                )
                if starts_word:
                    length = 1
                    for k in range(j + 1, self.width):
                        if self.structure[i][k]:
                            length += 1
                        else:
                            break
                    if length > 1:
                        self.variables.add(Variable(
                            i=i, j=j,
                            direction=Variable.ACROSS,
                            length=length
                        ))

        # Compute overlaps for each word
        # For any pair of variables v1, v2, their overlap is either:
        #    None, if the two variables do not overlap; or
        #    (i, j), where v1's ith character overlaps v2's jth character
        self.overlaps = dict()
        self.overlaps_positions = dict()
        for v1 in self.variables:
            for v2 in self.variables:
                if v1 == v2:
                    continue
                cells1 = v1.cells
                cells2 = v2.cells
                intersection = set(cells1).intersection(cells2)
                if not intersection:
                    self.overlaps[v1, v2] = None
                else:
                    intersection = intersection.pop()
                    self.overlaps[v1, v2] = (
                        cells1.index(intersection),
                        cells2.index(intersection)
                    )
                    for cell in cells1:
                        for cell_ in cells2:
                            if cell == cell_:
                                self.overlaps_positions[v1, v2] = cell
                                break

    def neighbors(self, var):
        """Given a variable, return set of overlapping variables."""
        return set(
            v for v in self.variables
            if v != var and self.overlaps[v, var]
        )

class CrosswordCreator():

    def __init__(self, crossword, do_random = False):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.back_track_count = 0
        self.memoized_back_track_count = 0
        self.states = []
        self.do_random = do_random
        self.memoization_cache = dict()
        self.t_revise_time = 0
        self.t_revise_called = 0
        self.consistency_elapsed_time = 0
        self.consistency_called = 0
        self.assignment = {}
        self.assignment_stack = []

        # setting up the domains for each of the variables
        self.domains = {
            var: [self.get_required_length_answers(var.length)]
            for var in self.crossword.variables
        }

    # enforcing the node consistency here
    def get_required_length_answers(self, ans_length):
        output = []
        for word in self.crossword.words:
            if len(word) == ans_length:
                output.append(word.upper())
        # random.shuffle(output)
        # output = output[:5000]
        # using set for the domains, do randomizing here is no advantage
        return set(output)  # lets get the domain answers in list format

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

    def write_info(self, message, filename = '/content/log_info.txt', filemode = 'a'):
        with open(filename, filemode) as f:
            f.write(message)

    def display_only_grid(self, assignment, block_size = 25, f_size = 15, filename = None):

        """
            Save crossword assignment to an image file.
        """
        cell_size = block_size
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGB",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("/content/gdrive/MyDrive/Crossword Generator/Roboto-Regular.ttf", f_size)
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
                             rect[0][1] + ((interior_size - h) / 2) - 5),
                            letters[i][j], fill="black", font=font
                        )
        if filename is not None:
            img.save(filename)
        return img

    def print_domain_info(self):
        '''
            print the variable information with its present domain size.
        '''
        for var in self.crossword.variables:
            print(f"Cell Position: ({var.i, var.j}) | Direction: {var.direction.upper()} | Length: {var.length} --> Domain Size: {len(self.domains[var][-1])}")

    ### here starts the main solving category
    def solve(self):
        """
            Enforce node and arc consistency, and then solve the CSP.
        """
        print("Before: Domain sizes Arc-Consistency (Pre-Processing Step): ")
        self.print_domain_info()
        self.ac3()
        print("\nAfter: Domain sizes Arc-Consistency (Pre-Processing Step): ")
        self.print_domain_info()

        # back-tracking starts here
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.crossword.variables:
            valid_words = set()
            for word in self.domains[variable][-1]:
                if len(word) == variable.length:
                    valid_words.add(word)
            self.domains[variable][-1] = valid_words

    def revise(self, x, y, forward_checking = False):
        """
            Make variable `x` arc consistent with variable `y`.
            To do so, remove values from `self.domains[x]` for which there is no
            possible corresponding value for `y` in `self.domains[y]`.

            Return True if a revision was made to the domain of `x`; return
            False if no revision was made.
        """

        start_t = time.time()
        revised = False
        overlap = self.crossword.overlaps[x, y]

        if overlap:
            y_chars = set(word[overlap[1]] for word in self.domains[y][-1])  # Use a set for faster membership tests
            x_domain = self.domains[x][-1]

            # Optimize: Use list comprehension for faster filtering
            x_domain = {word for word in x_domain if word[overlap[0]] in y_chars}

            if len(x_domain) < len(self.domains[x][-1]):
                revised = True
                if forward_checking:
                    self.domains[x].append(x_domain)
                else:
                    self.domains[x][-1] = x_domain

        end_t = time.time()
        self.t_revise_time += end_t - start_t
        self.t_revise_called += 1

        return revised

    def ac3(self, arcs=None, f_checking = False):
        if arcs is None:
            arcs = deque([(v1, v2) for v1 in self.crossword.variables for v2 in self.crossword.neighbors(v1)])
        else:
            arcs = deque(arcs)

        revised_arcs = set()

        while arcs:
            x, y = arcs.popleft()  # Efficient pop from the left

            # check if the arc has already been revised
            if (x, y) in revised_arcs:
                continue

            if self.revise(x, y, forward_checking = f_checking):
                if len(self.domains[x][-1]) == 0:
                    return False
                revised_arcs.add((x, y))
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.append((z, x))
                    revised_arcs.add((z, x))
        return True

    ## assignment_complete checkup - shortcut way
    # def assignment_complete(self, assignment):
    #     self.ASSIGNMENT_COUNT += 1
    #     self.states.append(assignment)
    #     if len(assignment.keys()) / len(self.crossword.variables) > 0.9:
    #         print(assignment)
    #         return True
    #     return len(assignment.keys()) == len(self.crossword.variables)

    # assigment_complete checkup - longcut way
    def assignment_complete(self, assignment):
        """
            Return True if `assignment` is complete (i.e., assigns a value to each
            crossword variable); return False otherwise.
        """
        # self.ASSIGNMENT_COUNT += 1
        self.back_track_count += 1
        self.states.append(assignment.copy())
        # if len(assignment.keys()) / len(self.crossword.variables) >= 9.0:
        #     return True

        complete = True
        vars_in_assignment = set(var for var in assignment)
        # Checking if all vars in the crossword has been assigned
        if vars_in_assignment != self.crossword.variables:
            complete = False
        for var in assignment:
            # making sure no var is empty
            # assert isinstance(self.assignment[var], str)
            if not assignment[var]:
                complete = False
        return complete

    # phind AI
    def consistent(self, assignment):
        """
            Return True if `assignment` is consistent (i.e., words fit in crossword
            puzzle without conflicting characters); return False otherwise.
        """
        start_t = time.time()
        values = set()
        for var, word in assignment.items():
            if word in values or len(word) != var.length:
                end_t = time.time()
                self.consistency_elapsed_time += end_t - start_t
                self.consistency_called += 1
                return False
            values.add(word)
            for neighbor in self.crossword.neighbors(var):
                overlap = self.crossword.overlaps[var, neighbor]
                if neighbor in assignment:
                    if assignment[var][overlap[0]] != assignment[neighbor][overlap[1]]:
                        end_t = time.time()
                        self.consistency_elapsed_time += end_t - start_t
                        self.consistency_called += 1
                        return False
        end_t = time.time()
        self.consistency_elapsed_time += end_t - start_t
        self.consistency_called += 1
        return True

    def order_domain_values(self, var, assignment, temp_var_domain):
        # start_t = time.time()
        values_penalty = Counter()
        for neighbor in self.crossword.neighbors(var):
            if neighbor not in assignment:
                overlap = self.crossword.overlaps[var, neighbor]
                neighbor_list = [value[overlap[1]] for value in list(self.domains[neighbor][-1])]

                for value in temp_var_domain:
                    letter_to_be_searched = neighbor_list.count(value[overlap[0]])
                    values_penalty[value] += len(neighbor_list) - letter_to_be_searched

        priority_queue = [(-values_penalty[value], value) for value in temp_var_domain]
        heapq.heapify(priority_queue)
        # end_t = time.time()
        # print("Ordering the domain values: ", end_t - start_t)
        return [value for _, value in priority_queue]


    # def select_unassigned_variable(self, assignment):
    #     """
    #         Ordering:
    #           Return an unassigned variable not already part of `assignment`.
    #           Choose the variable with the minimum number of remaining values - MRV
    #           in its domain. If there is a tie, choose the variable with the highest
    #           degree. If there is a tie, any of the tied variables are acceptable
    #           return values.
    #     """
    #     var_penalty = {}
    #     for var in self.crossword.variables:
    #         if var not in assignment:
    #             var_penalty[var] = len(self.domains[var][-1])
    #             # var_penalty[var] = var.length
    #             # var_penalty[var] = len(self.crossword.neighbors(var))
    #     vars = sorted(var_penalty, key= lambda v: var_penalty[v], reverse = True)

    #     # if len(vars) > 1 and var_penalty[vars[0]] == var_penalty[vars[1]]:
    #     #     # Check number of neighbors and return highest degree
    #     #     if len(self.crossword.neighbors(vars[0])) < len(self.crossword.neighbors(vars[1])):
    #     #         return vars[1]
    #     return vars[0]

    # smart-choosing approach
    def select_unassigned_variable(self, assignment):
        ## DEBUG purpose
        if len(assignment.keys()) == len(self.crossword.variables):
            return None

        # if it is choosing unassigned variable for the firs time then
        var_penalty = {}
        if len(assignment.keys()) == 0:
            for var in self.crossword.variables:
                var_penalty[var] = len(self.crossword.neighbors(var))
            vars = sorted(var_penalty, key = lambda v: var_penalty[v], reverse = True)

            return vars[0] # -> if choosing for the first time, then choose the one with the maximum attached neighbors

        else:
            # lets find the slot that has the most engaging assignment at first,
            # if not engaging assignment is found then look for the next in the assignment stack
            for var in self.crossword.variables:
                var_penalty[var] = [0, 0]
                if var not in assignment:
                    for neighbor in self.crossword.neighbors(var):
                        if neighbor in assignment:
                            var_penalty[var][0] += 1
                            var_penalty[var][1] = var_penalty[var][0] / var.length


            for var, (count, percentage) in var_penalty.items():
                if count > 1:
                    vars = sorted(var_penalty, key = lambda v: var_penalty[v][1], reverse = True)
                    return vars[0]

            var_penalty = {}
            for var_assignment in self.assignment_stack:
                var_penalty = {}
                for neighbor in self.crossword.neighbors(var_assignment):
                    if neighbor not in assignment:
                        var_penalty[neighbor] = len(self.crossword.neighbors(neighbor))
                if len(var_penalty.keys()) != 0:
                    vars = sorted(var_penalty, key = lambda v: var_penalty[v], reverse = True)
                    return vars[0]

            # previous_var = self.assignment_stack[-1]
            # for neighbor in self.crossword.neighbors(previous_var)
            #     if neighbor not in assignment:
            #         var_penalty[neighbor] = len(self.crossword.neighbors(neighbor))


            if len(var_penalty.keys()) == 0:
                for var in self.crossword.variables:
                    if var not in assignment:
                        var_penalty[var] = len(self.crossword.neighbors(var))
                vars = sorted(var_penalty, key = lambda v: var_penalty[v], reverse = True)
                return vars[0]

            else:
                vars = sorted(var_penalty, key = lambda v: var_penalty[v], reverse = True)
                return vars[0]

    # random-choosing approach
    # def select_unassigned_variable(self, assignment):
    #     unassigned_variable_list = []
    #     for var in self.crossword.variables:
    #         if var not in assignment:
    #             unassigned_variable_list.append(var)

    #     return random.choice(unassigned_variable_list)


    # @lru_cache(None)
    def backtrack(self, assignment, assigned_var = None):
        """
            Using Backtracking Search, take as input a partial assignment for the
            crossword and return a complete assignment if possible to do so.

            `assignment` is a mapping from variables (keys) to words (values).

            If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            print(assignment)
            return assignment  # base case

        # lets have some caching done here
        assignment_key = frozenset(assignment.items())
        if assignment_key in self.memoization_cache:
            # self.ASSIGNMENT_COUNT += 1
            self.memoized_back_track_count += 1
            return self.memoization_cache[assignment_key]

        # selecting a new variable
        var = self.select_unassigned_variable(assignment)

        did_shorten_domain = False

        temp_domain_var = self.domains[var][-1].copy()

        # print(var, "Before: ", len(temp_domain_var))
        self.write_info(message = f"{var} | Before: {len(temp_domain_var)}\n")
        if len(assignment.keys()) > 0:
            for variable in self.crossword.variables:
                if var != variable and variable in assignment.keys():
                    overlap = self.crossword.overlaps[var, variable]
                    if overlap:
                        # print("Inside the overlap function!!!")
                        ref_cross_section_word = assignment[variable]
                        ref_intersection_letter = ref_cross_section_word[overlap[1]]

                        # Filter the words in the domain of var
                        did_shorten_domain = True
                        temp_domain_var = {word for word in temp_domain_var if word[overlap[0]] == ref_intersection_letter}

        # print(var, "After: ", len(temp_domain_var))
        self.write_info(message = f"{var} | After: {len(temp_domain_var)}\n")

        if did_shorten_domain:
            if len(temp_domain_var) == 0:
                # print("The temporary domain var is Empty")
                self.memoization_cache[assignment_key] = None
                return None
            else:
                self.domains[var].append(temp_domain_var)

        revised_neighbor_list = []
        # lets insert the ac3-arc consistency in the back-track recursive loop
        arc_list = deque([(neighbor, var) for neighbor in self.crossword.neighbors(var)])
        revised_arcs = set()

        while arc_list:
            x, y = arc_list.pop()

            if (x, y) in revised_arcs:
                continue

            if self.revise(x, y, forward_checking = True):
                print("did some arc-consistency suff")
                print(len(self.domains[x][-1]))
                revised_neighbor_list.append(x)

                for z in self.crossword.neighbors(x) - {y}:
                    arc_list.append((z, x))
                    revised_arcs.add((z, x))

        for var_ in self.crossword.variables:
            if var_ not in self.assignment:
                if len(self.domains[var_][-1]) == 0:
                    for n in revised_neighbor_list:
                        self.domains[n].pop()
                    revised_neighbor_list = []
                    print("Getting out because the arc_list value has banished")
                    return None


        # print("After: ", len(self.domains[var][-1]))

        # lets introduce the randomness in iterating the values of the assigned variabel
        if self.do_random:
            shuffled_curr_domain = list(self.domains[var][-1].copy())
            random.shuffle(shuffled_curr_domain)

        domain_values = shuffled_curr_domain if self.do_random else list(self.domains[var][-1])
        # print(var, len(domain_values))
        new_assignment = assignment.copy()
        for value in domain_values:
            # print(var, value)
            self.write_info(message = f"{var} || Answer: {value}\n")
            # new_assignment[var] = value
            assignment[var] = value
            if self.consistent(assignment):
                self.assignment_stack.append(var)
                result = self.backtrack(assignment, var)

                if result is not None:
                    self.memoization_cache[assignment_key] = result
                    return result

                self.assignment_stack.pop()

        for n in revised_neighbor_list:
            self.domains[n].pop()

        if did_shorten_domain:
            self.domains[var].pop()

        self.memoization_cache[assignment_key] = None
        # if assigned_var is not None:
        assignment.pop(var)
        # self.assignment_stack.pop()
        return None