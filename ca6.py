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
# 
# 1. Add a method `norm(p)` to your `Vec` class so that if `u` is a `Vec` object, then `u.norm(p)` returns the $L_p$-norm of vector `u`.  Recall that the $L_p$-norm of an $n$-dimensional vector $\overrightarrow{u}$ is given by, $||u||_p = \left( \sum_{i = 1}^{n} |u_i|^p\right)^{1/p}$.  Input `p` should be of the type `int`.  The output norm should be of the type `float`.
# 2. Add a method `rank()` to your `Matrix` class so that if `A` is a `Matrix` object, then `A.rank()` returns the rank of `A`.

# %%
# COPY-PASTE YOUR Matrix AND Vec CLASSES TO THIS CELL. 

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
    # todo
    pass

def solve_np(A, b):
    #todo
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
import enum

class GaussSolvers(enum.Enum):
    np = 0
    pp = 1
    tp = 2
    
    
def solve(A, b, solver = GaussSolvers.np):
    if solver == GaussSolvers.np:
        return solve_np(A, b)
    elif solver == GaussSolvers.pp:
        return solve_pp(A, b)
    elif solver == GaussSolvers.tp:
        return solve_tp(A, b)

# %%
"""TESTER CELL #1"""
A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

x = Vec([2.3, -4.1, 5.7]) # this is the true solution

b = A * x

x_np = solve(A, b)
x_pp = solve(A, b, GaussSolvers.pp)
x_tp = solve(A, b, GaussSolvers.tp)

epsilon_np = x_np - x
epsilon_pp = x_pp - x
epsilon_tp = x_tp - x

error1_np = epsilon_np.norm(1)
error2_np = epsilon_np.norm(2)

print("-"*20)
print("No Pivoting Solution:", x_np)
print("True solution:", x)

print("Errors in Gaussian Elimination Without Pivoting")
print("L1-norm error:", error1_np)
print("L2-norm error:", error1_np)
print()

print("-"*20)
print("Partial Pivoting Solution:", x_pp)
print("True solution:", x)

error1_pp = epsilon_pp.norm(1)
error2_pp = epsilon_pp.norm(2)

print("Errors in Gaussian Elimination With Partial Pivoting")
print("L1-norm error:", error1_pp)
print("L2-norm error:", error1_pp)
print()

print("-"*20)
print("Total Pivoting Solution:", x_tp)
print("True solution:", x)

error1_tp = epsilon_tp.norm(1)
error2_tp = epsilon_tp.norm(2)

print("Errors in Gaussian Elimination With Total Pivoting")
print("L1-norm error:", error1_tp)
print("L2-norm error:", error1_tp)


# %%
"""TESTER CELL #2"""

A = Matrix([[1, 2, 3], [2, 4, 6]])

b = Vec([6, -12]) 

x_np = solve(A, b)
x_pp = solve(A, b, GaussSolvers.pp)
x_tp = solve(A, b, GaussSolvers.tp)

print("-"*20)
print("No Pivoting Solution:", x_np)
print("Expected: None")

print("-"*20)
print("Partial Pivoting Solution:", x_pp)
print("Expected: None")

print("-"*20)
print("Total Pivoting Solution:", x_tp)
print("Expected: None")

# %%
"""TESTER CELL #3"""

# Test one of the examples from lecture that had infinitely-many solutions

# %% [markdown]
# 
# ------------------------------------------------
# 
# #### Problem 5
# 
# Implement the method `is_independent(S)` that returns `True` if the set `S` of `Vec` objects is linearly **independent**, otherwise returns `False`.

# %%
def is_independent(S):
    #todo
    pass

# %%
"""TESTER CELL"""

S1 = {Vec([1, 2]), Vec([2, 3]), Vec([3, 4])}

print("S1 is Independent:", is_independent(S1))
print("Expected: False")

S2 = {Vec([1, 1, 1]), Vec([1, 2, 3]), Vec([1, 3, 6])}

print("S2 is Independent:", is_independent(S2))
print("Expected: True")




