# %% [markdown]
# # CECS 229 Coding Assignment #6
# 
# #### Due Date: 
# 
# Sunday, 4/24 @ 11:59 PM
# 
# #### Submission Instructions:
# 
# To receive credit for this assignment you must submit the following by the due date:
# 
# 1. **To the BB Dropbox Folder:** this completed .ipynb file
# 
# 2. **To CodePost:** this file converted to a Python script named `ca6.py`
# 
# #### Objectives:
# 
# 1. Apply Gaussian Elimination, with and without pivoting, to help solve the system $A \overrightarrow{x} = \overrightarrow{b}$.
# 2. Use Lp -norm to calculate the error in a solution given by applying any of the Gaussian elimination algorithms.
# 3. Use the RREF of the augmented matrix for the system $A \overrightarrow{x} = \overrightarrow{b}$ to determine if it has one solution, no solution, or infinitely-many solutions.
# 4. Determine the number of free variables that the system $A \overrightarrow{x} = \overrightarrow{b}$ has if it has infinitely-many solutions.
# 5. Determine whether a set of column-vectors is linearly dependent by forming and solving a system of the form $A \overrightarrow{x} = \overrightarrow{0}$.
# -----------------------------------------
# 
# 

# %% [markdown]
# #### Problem 1
# 
# Copy-paste your implemented `Matrix` and `Vec` classes to the next cell.  Then, complete the following tasks:
# 1. Add a method `norm(p)` to your `Vec` class so that if `u` is a `Vec` object, then `u.norm(p)` returns the $L_p$-norm of vector `u`.  Recall that the $L_p$-norm of an $n$-dimensional vector $\overrightarrow{u}$ is given by, $||u||_p = \left( \sum_{i = 1}^{n} |u_i|^p\right)^{1/p}$.  Input `p` should be of the type `int`.  The output norm should be of the type `float`.
# 2. Add a method `rank()` to your `Matrix` class so that if `A` is a `Matrix` object, then `A.rank()` returns the rank of `A`.

# %%
# COPY-PASTE YOUR Matrix AND Vec CLASSES TO THIS CELL. 
import math

class Matrix:
    
    def __init__(self, matrix = []):  #FIXME: Add necessary parameters and default values
        self.colsp = []
        self.rowsp = []

        for i, row in enumerate(matrix):
            self.rowsp.append(row)

        for i in range(len(matrix[0])):
            temp = []
            for j in range(len(matrix)):
                temp.append(matrix[j][i])
            self.colsp.append(temp)


        
    def set_col(self, j, u):
        j = j - 1
        if len(u) != len(self.colsp[j]):
            print("Incompatible column length")
            raise ValueError
        
        self.colsp[j] = u

        for i, row in enumerate(self.rowsp):
            row[j] = u[i]
    
    def set_row(self, i, v):
        i = i - 1
        if len(v) != len(self.rowsp[i]):
            print("Incompatible column length")
            raise ValueError
        
        self.rowsp[i] = v

        for j, col in enumerate(self.colsp):
            col[i] = v[j]
    
    def set_entry(self, i, j, x):
        i = i - 1
        j = j - 1
        
        self.rowsp[i][j] = x
        self.colsp[j][i] = x

    def get_col(self, j):
        j = j - 1
        return self.colsp[j]
    
    def get_row(self, i):
        i = i - 1
        return self.rowsp[i]

    def get_entry(self, i, j):
        i = i - 1
        j = j - 1
        return self.rowsp[i][j]

    def col_space(self):
        return self.colsp
    
    def row_space(self):
        return self.rowsp
    
    def get_diag(self, k):
        ans = []
        row_counter = 0
        if k < 0:
            row_counter = abs(k)

        col_counter = 0
        
        for i in range(k, len(self.rowsp)):

            if k < 0:
                ans.append(self.get_entry(row_counter + 1, col_counter + 1))

                row_counter = row_counter + 1
                col_counter = col_counter + 1
                if row_counter == len(self.rowsp):
                    break

            elif k == 0:
                ans.append(self.get_entry(i + 1, i + 1))
            elif k > 0:
                ans.append(self.get_entry(row_counter + 1, i + 1))
                row_counter = row_counter + 1
        
        return ans
    
    def __add__(self, other):
        if len(other.row_space()) != len(self.rowsp) or len(other.col_space()) != len(self.colsp):
            print("Incompatible column length")
            raise ValueError
        
        ans = []
        for i in range(0, len(self.rowsp)):
            temp = []
            for j in range(0, len(self.rowsp[i])):
                temp.append(self.rowsp[i][j] + other.row_space()[i][j])

            ans.append(temp)


        return Matrix(ans)

    
    def __sub__(self, other):
        if len(other.row_space()) != len(self.rowsp) or len(other.col_space()) != len(self.colsp):
            print("Incompatible column length")
            raise ValueError
        
        ans = []
        for i in range(0, len(self.rowsp)):
            temp = []
            for j in range(0, len(self.rowsp[i])):
                temp.append(self.rowsp[i][j] - other.row_space()[i][j])

            ans.append(temp)


        return Matrix(ans)
        
    def __mul__(self, other):  
        ans = []

        # A: 3x2 , B: 2x2

        print(self.rowsp)

        if isinstance(other, float) or isinstance(other, int):
            for i in range(0, len(self.rowsp)):
                temp = []
                for j in range(0, len(self.rowsp[i])):
                    temp.append(self.rowsp[i][j] * other)
                print(temp)
                ans.append(temp)

        elif isinstance(other, Matrix):

            
            if len(other.row_space()) != len(self.colsp):
                print("Incompatible length")
                raise ValueError
            
            print(other.colsp)
            ans = []
            for i in range(0, len(self.rowsp)):
                temp = []
                for j in range(0, len(other.colsp)):
                    sum = 0
                    for k in range(0, len(other.rowsp)):
                        sum = sum + (self.rowsp[i][k] * other.rowsp[k][j])
                    temp.append(sum)

                ans.append(temp)



        elif isinstance(other, Vec):
            ans = []
            for i in range(0, len(self.rowsp)):
                sum = 0
                for j in range(0, len(self.rowsp[i])):
                    sum = sum + (self.rowsp[i][j] * other.elements[j])
                ans.append(sum)
            return Vec(ans)
             
        else:
            print("ERROR: Unsupported Type.")
            raise ValueError
            
        return Matrix(ans)
    
    def __rmul__(self, other):  
        ans = []
        if isinstance(other, float) or isinstance(other, int):

            for i in range(0, len(self.rowsp)):
                temp = []
                for j in range(0, len(self.rowsp[i])):
                    temp.append(self.rowsp[i][j] * other)
                print(temp)
                ans.append(temp)
        else:
            print("ERROR: Unsupported Type.")
            raise ValueError
            
        return Matrix(ans)
    
    def __str__(self):
        """prints the rows and columns in matrix form """
        ans = ''
        for row in self.rowsp:
            ans += str(row) + "\n"
        return ans
        
    def __eq__(self, other):
        """overloads the == operator to return True if 
        two Matrix objects have the same row space and column space"""
        this_rows = self.row_space()
        other_rows = other.row_space()
        this_cols = self.col_space()
        other_cols = other.col_space()
        return this_rows == other_rows and this_cols == other_cols

    def __req__(self, other):
        """overloads the == operator to return True if 
        two Matrix objects have the same row space and column space"""
        this_rows = self.row_space()
        other_rows = other.row_space()
        this_cols = self.col_space()
        other_cols = other.col_space()
        return this_rows == other_rows and this_cols == other_cols

    
    def rank(self):
        rank = 0
        zero_vector = [0 for i in range(len(self.rowsp[0]))]
        for row in self.rowsp:
            if row != zero_vector:
                rank += 1
        return rank
    


    # This code is not tested yet, may contain a lot of bugs
    def reduce_scalar_mult(self):
        # Use to construct reduce matrix, the first number is the larger row
        scalar_mult_row_table = {} 

        # Check for rows that are scalar multiples
        for i, row_curr in enumerate(self.rowsp):
            for j, row_not_curr in enumerate(self.rowsp[i:], i):
                if row_not_curr[0] % row_curr[0] == 0:
                    mult = row_not_curr[0] / row_curr[0]
                    num_of_scalar_mult = 0
                    for j, col in enumerate(row_curr):
                        if row_not_curr[j] % row_curr[j] != mult:
                            break
                        else:
                            num_of_scalar_mult += 1
                    if num_of_scalar_mult == len(row_curr):
                        scalar_mult_row_table[j] = i
                        
                elif row_curr[0] % row_not_curr[0]:
                    mult = row_curr[0] / row_not_curr[0]
                    num_of_scalar_mult = 0
                    for j, col in enumerate(row_curr):
                        if row_curr[j] % row_not_curr[j] != mult:
                            break
                        else:
                            num_of_scalar_mult += 1
                    if num_of_scalar_mult == len(row_curr):
                        scalar_mult_row_table[i] = j

        # Constructing reduce matrix
        ans = self.rowsp
        num_of_removed_items = 0
        removed_items = []
        for pair in scalar_mult_row_table.items():
            if pair[0] not in removed_items:
                ans.pop(pair[0] - num_of_removed_items)
                num_of_removed_items += 1
                removed_items.append(pair[0])
        
        return Matrix(ans)





                    


# From Assignment 4
class Vec:
    def __init__(self, contents = []):
        """
        Constructor defaults to empty vector
        INPUT: list of elements to initialize a vector object, defaults to empty list
        """
        self.elements = contents
        return
    
    def __abs__(self):
        """
        Overloads the built-in function abs(v)
        returns the Euclidean norm of vector v
        """
        ans = math.sqrt(sum([pow(i,2) for i in self.elements]))
        return ans
        
    def __add__(self, other):
        """Overloads the + operator to support Vec + Vec
         raises ValueError if vectors are not same length
        """
        if len(self.elements) != len(other.elements):
            raise ValueError
        i = 0
        ans = []
        while i < len(self.elements):
            ans.append(self.elements[i] + other.elements[i])
            i = i + 1

        return Vec(ans)
            
    
    def __sub__(self, other):
        """
        Overloads the - operator to support Vec - Vec
        Raises a ValueError if the lengths of both Vec objects are not the same
        """
        if len(self.elements) != len(other.elements):
            raise ValueError
        i = 0
        ans = []
        while i < len(self.elements):
            ans.append(self.elements[i] - other.elements[i])
            i = i + 1

        return Vec(ans)
    
    
    
    def __mul__(self, other):
        """Overloads the * operator to support 
            - Vec * Vec (dot product) raises ValueError if vectors are not same length in the case of dot product
            - Vec * float (component-wise product)
            - Vec * int (component-wise product)
            
        """
        if isinstance(other) == Vec: #define dot product
            if len(self.elements) != len(other.elements):
                raise ValueError
            i = 0
            ans = 0
            while i < len(self.elements):
                ans = ans + (self.elements[i] * other.elements[i])
                i = i + 1

            return ans
            
        elif isinstance(other) == float or isinstance(other) == int: #scalar-vector multiplication
            return Vec([num * other for num in self.elements])
            
    
    def __rmul__(self, other):
        """Overloads the * operation to support 
            - float * Vec
            - int * Vec
        """
        return Vec([num * other for num in self.elements])
    

    
    def __str__(self):
        """returns string representation of this Vec object"""
        return str(self.elements) # does NOT need further implementation

    def norm(self, p):
        #p = int
        #out = float
        #norm = 
        sum = 0.0
        if isinstance(p) == Vec:
            for element in p:
                sum += pow(element, 2)
            return math.sqrt(sum)

        

# %% [markdown]
# #### Problem 2
# 
# 1. Implement a helper function called `_rref(A, b)` that applies Gaussian Elimination ***without pivoting*** to return the Reduced-Row Echelon Form of the augmented matrix formed from `Matrix` object `A` and `Vec` object `b`.  The output must be of the type `Matrix`.
# 2. Implement the function `solve_np(A, b)` that uses `_rref(A)` to solve the system $A \overrightarrow{x} = \overrightarrow{b}$.  The input `A` is of the type `Matrix` and `b` is of the type `Vec`.
#     - If the system has a unique solution, it returns the solution as a `Vec` object.  
#     - If the system has no solution, it returns `None`. 
#     - If the system has infinitely many solutions, it returns the number of free variables (`int`) in the solution.

# %%


def _rref(A, b):
    aug_matrix = []
    # Forming the augmented matrix
    for i, row in enumerate(A.rowsp):
        temp = row
        temp.append(b.elements[i])
        aug_matrix.append(temp)

    print(aug_matrix)
    # Sort row
    maxedAug = []

    while(len(aug_matrix) > 0):
        maxRow = aug_matrix[0]
        for row in aug_matrix:
            if row[0] > maxRow[0]:
                maxRow = row
        maxedAug.append(maxRow)
        aug_matrix.remove(maxRow)

    aug_matrix = maxedAug

    print(aug_matrix)

    # Doing gaussian elimination
    pivot = aug_matrix[0][0]
    pivot_index = 0
    for i, row in enumerate(aug_matrix):
        if i != 0:
            alpha = row[0]/pivot
            for j, col in enumerate(row[pivot_index:], i):
                col = col - (alpha/pivot) * (aug_matrix[pivot_index][j])
            pivot = aug_matrix[i][i]
            pivot_index = i
    
    return Matrix(aug_matrix)

m = Matrix([[0, 0, 2],
           [0, 3, 1],
           [3, 4, 1]])
           
vec = Vec([1, 1, 1])

print(_rref(m, vec))


def solve_np(A, b):
    aug_matrix = []
    # Forming the augmented matrix
    for i, row in enumerate(A):
        temp = row
        temp = temp.append([b[i]])
        aug_matrix.append(temp)

        aug_matrix = Matrix(aug_matrix)
    
    if A.rank() < aug_matrix.rank():
        return None
    elif A.rank() == aug_matrix.rank():
        pass
    else:
        pass

# %% [markdown]
# #### Problem 3
# 
# 1. Implement a helper function called `_rref_pp(A, b)` that applies Gaussian Elimination ***with partial pivoting*** to return the Reduced-Row Echelon Form of the augmented matrix formed from `Matrix` object `A` and `Vec` object `b`.  The output must be of the type `Matrix`.
# 2. Implement the function `solve_pp(A, b)` that uses `_rref_pp(A, b)` to solve the system $A \overrightarrow{x} = \overrightarrow{b}$.  The input `A` is of the type `Matrix` and `b` is of the type `Vec`.  
#     - If the system has a unique solution, it returns the solution as a `Vec` object.  
#     - If the system has no solution, it returns `None`. 
#     - If the system has infinitely many solutions, it returns the number of free variables (`int`) in the solution.

# %%
def _rref_pp(A, b):
    # todo
    pass

def solve_pp(A, b):
    #todo
    pass

# %% [markdown]
# #### Problem 4
# 
# 1. Implement a helper function called `_rref_tp(A, b)` that applies Gaussian Elimination ***with total pivoting*** to return the Reduced-Row Echelon Form of the augmented matrix formed from `Matrix` object `A` and `Vec` object `b`.  The output must be of the type `Matrix`. 
# 2. Implement the function `solve_tp(A, b)` that uses `_rref_tp(A)` to solve the system $A \overrightarrow{x} = \overrightarrow{b}$.  The input `A` is of the type `Matrix` and `b` is of the type `Vec`. 
#     - If the system has a unique solution, it returns the solution as a `Vec` object.  
#     - If the system has no solution, it returns `None`. 
#     - If the system has infinitely many solutions, it returns the number of free variables (`int`) in the solution.

# %%
def _rref_tp(A):
    # todo
    pass

def solve_tp(A, b):
    #todo
    pass

# %% [markdown]
# #### Master function
# 
# The following function is the master function that will be called by the CodePost tester.  It will be fully functional once you have completed Problems 1 - 4.  No edits are necessary.

# %%
# import enum

# class GaussSolvers(enum.Enum):
#     np = 0
#     pp = 1
#     tp = 2
    
    
# def solve(A, b, solver = GaussSolvers.np):
#     if solver == GaussSolvers.np:
#         return solve_np(A, b)
#     elif solver == GaussSolvers.pp:
#         return solve_pp(A, b)
#     elif solver == GaussSolvers.tp:
#         return solve_tp(A, b)

# # %%
# """TESTER CELL #1"""
# A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# x = Vec([2.3, -4.1, 5.7]) # this is the true solution

# b = A * x

# x_np = solve(A, b)
# x_pp = solve(A, b, GaussSolvers.pp)
# x_tp = solve(A, b, GaussSolvers.tp)

# epsilon_np = x_np - x
# epsilon_pp = x_pp - x
# epsilon_tp = x_tp - x

# error1_np = epsilon_np.norm(1)
# error2_np = epsilon_np.norm(2)

# print("-"*20)
# print("No Pivoting Solution:", x_np)
# print("True solution:", x)

# print("Errors in Gaussian Elimination Without Pivoting")
# print("L1-norm error:", error1_np)
# print("L2-norm error:", error1_np)
# print()

# print("-"*20)
# print("Partial Pivoting Solution:", x_pp)
# print("True solution:", x)

# error1_pp = epsilon_pp.norm(1)
# error2_pp = epsilon_pp.norm(2)

# print("Errors in Gaussian Elimination With Partial Pivoting")
# print("L1-norm error:", error1_pp)
# print("L2-norm error:", error1_pp)
# print()

# print("-"*20)
# print("Total Pivoting Solution:", x_tp)
# print("True solution:", x)

# error1_tp = epsilon_tp.norm(1)
# error2_tp = epsilon_tp.norm(2)

# print("Errors in Gaussian Elimination With Total Pivoting")
# print("L1-norm error:", error1_tp)
# print("L2-norm error:", error1_tp)


# # %%
# """TESTER CELL #2"""

# A = Matrix([[1, 2, 3], [2, 4, 6]])

# b = Vec([6, -12]) 

# x_np = solve(A, b)
# x_pp = solve(A, b, GaussSolvers.pp)
# x_tp = solve(A, b, GaussSolvers.tp)

# print("-"*20)
# print("No Pivoting Solution:", x_np)
# print("Expected: None")

# print("-"*20)
# print("Partial Pivoting Solution:", x_pp)
# print("Expected: None")

# print("-"*20)
# print("Total Pivoting Solution:", x_tp)
# print("Expected: None")

# # %%
# """TESTER CELL #3"""

# # Test one of the examples from lecture that had infinitely-many solutions

# # %% [markdown]
# # 
# # ------------------------------------------------
# # 
# # #### Problem 5
# # 
# # Implement the method `is_independent(S)` that returns `True` if the set `S` of `Vec` objects is linearly **independent**, otherwise returns `False`.

# # %%
# def is_independent(S):
#     #todo
#     pass

# # %%
# """TESTER CELL"""

# S1 = {Vec([1, 2]), Vec([2, 3]), Vec([3, 4])}

# print("S1 is Independent:", is_independent(S1))
# print("Expected: False")

# S2 = {Vec([1, 1, 1]), Vec([1, 2, 3]), Vec([1, 3, 6])}

# print("S2 is Independent:", is_independent(S2))
# print("Expected: True")




