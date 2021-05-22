import numpy as np
import scipy.sparse as sp
import scipy.integrate as ode
import time

# Container for a coarse-graining method based on partitioning
class Method:
    # partition is an array which contains the class in the partition for each state
    def __init__(self, partition):
        L, R, P = partitioning_projector(partition)
        self.L = L
        self.R = R
        self.P = P
        self.dim = R.shape[0]
        self.A = None
        self.solution = None
        self.error = None
        self.error_norm = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.qsd = None
        self.qsd_val_shift = None
        self.qsd_vec_shift = None
        self.ePvQ = None

    # Initializes the laplacian A. Only the coarse-grained laplacian is saved.
    # For large state-spaces A should be a scipy.sparse matrix.
    def set_matrix(self, A):
        self.A = self.R@A@self.L

    # Resets the method to another partition
    def set_partition(self, partition):
        self.__init__(self, partition)

    # Computes the solution of the coarse-grained system. x0 is the initial condition on the full state-space.
    # The solution is given on [0,T] in k time frames.
    def solve_system(self, x0, T, k):
        if self.A is None:
            raise NullError('The matrix of the method is undefined!')
        x0_ = x0 @ self.L
        self.solution = solve_ODEs(x0_, self.A, T, k)

    # Computes the error against the exact solution. self.error is the error of each state in each time frame.
    # self.error_norm contains the euclidean norm of the errors of all states per time frame.
    def calc_error(self, exact_Solution):
        if self.solution is None:
            raise NullError('Solution requested before it was computed!')
        self.error = np.abs(exact_Solution - self.lifted_solution())
        self.error_norm = np.linalg.norm(self.error, axis=1)

    # Lift the solution of the coarse-grained system onto the full state-space to be comparable with other solutions.
    def lifted_solution(self):
        if self.solution is None:
            raise NullError('Solution requested before it was computed!')
        return self.solution @ self.R

    # Computes the dominant eigenpair structure of the coarse grained system. val and vec are the three dominant
    # eigenvalues and -vectors of the exact system.
    # self.eigenvalues and self.eigenvectors saves the three dominant eigenpairs of the coarse-grained system, lifted
    # to the full state-space.
    # self.qsd_val_shift and self.qsd_vec_shift save the absolute difference of the second leading eigenvalue and the
    # norm of the difference between the second leading eigenvector of the coarse-grained and exact system.
    # self.qsd is the second leading eigenvector on the non-zero states, normalized to sum to 1.
    # self.qsd_eff_shift is the norm between the QSD of the coarse-grained and exact system on the non-zero states,
    # normalized such that they sum to 1.
    # self.ePvQ is the euclidean distance between v and the image of P.
    def calc_leading_ev(self, val, vec):
        if self.A is None:
            raise NullError('The matrix of the method is undefined!')
        e,v = sp.linalg.eigs(self.A.transpose(), 3, which='SM')
        order = np.argsort(np.abs(e))
        e = e[order]
        v = v.transpose()[order]
        v = v @ self.R
        v[0] = v[0] * np.sign(v[0][0].real)
        v[1] = -v[1] * np.sign(v[1][0].real)
        if np.abs(v[0][0] - 1) > 1e-12 or np.abs(e[0]) > 1e-12:
            raise ValueError('Leading Eigenpair has an unexpected value!')
        self.eigenvalues = e
        self.eigenvectors = v
        self.qsd_val_shift = np.abs(e[1] - val[1])
        self.qsd_vec_shift = np.linalg.norm(v[1] - vec[1])
        self.qsd = -v[1] / v[1][0]
        scaled_vec = -vec[1] / vec[1][0]
        self.qsd_eff_shift = np.linalg.norm(self.qsd - scaled_vec)
        self.ePvQ = np.linalg.norm(vec[1] - vec[1] @ self.P)


# Returns the sizes of the classes of a given partition
def class_sizes(partition):
    m = np.max(partition) + 1
    sizes = np.zeros(m, dtype=int)
    for a in range(m):
        sizes[a] = len(np.where(partition==a)[0])
    return sizes


# Creates the projector P=LR, which obeys the given partitioning. L consist only of 0's and 1's.
# partitioning: An array of length n containing the values {0,...,m-1}, which denote the respective class of a state.
def partitioning_projector(partinion):
    m = np.max(partinion) + 1    # number of classes
    n = len(partinion)           # number of states

    sizes = class_sizes(partinion)

    row = np.array(list(range(n)))
    col = partinion
    data = np.ones(n)
    L = sp.csc_matrix((data,(row, col)), (n, m))

    row = partinion
    col = np.array(list(range(n)))
    data = 1 / sizes[partinion]
    R = sp.csc_matrix((data,(row, col)), (m, n))

    P = L@R

    return L, R, P


# Solves the linear system of ODE's defined by A with initial value x0.
# The solution is given on a grid of n points on [0, T] as a an kxn-array, where n is the dimension of the system---
def solve_ODEs(x0, A, T, k):
    def func(x, t):
        return x @ A

    t = np.linspace(0, T, k)
    start = time.time()
    solution = ode.odeint(func, x0, t)
    end = time.time()

    print('Integration-time: {}'.format(end - start))

    return solution


# Evaluates the solutions are errors of the methods given. Returns the exact solution.
# methods: list of objects of the class Method.
# x0: initial condition of the system.
# A: preferably sparse laplacian of the system
# The solutions are evaluated on [0, T] in k time frames.
def compare_methods(methods, x0, A, T, k):
    n = A.shape[0]
    m = len(methods)

    exact_Solution = solve_ODEs(x0, A, T, k)

    for i in range(m):
        M = methods[i]
        M.set_matrix(A)
        M.solve_system(x0, T, k)
        M.calc_error(exact_Solution)

    return exact_Solution





