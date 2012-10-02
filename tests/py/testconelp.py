#
# This is copied from CVXOPT examples and modified to be used as test reference
# for corresponding Go program.
#
# The small linear cone program of section 8.1 (Linear cone programs).

import sys
from cvxopt import matrix, solvers
from cvxopt import misc, blas
import localcones
import helpers


def solve(opts):
    c = matrix([-6., -4., -5.])

    G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,   7.,  
                  -5.,   1.,  -5.,   1.,  -7.,   1.,   -7.,  -4.],
                [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,   3.,  
                  13.,  -6.,  13.,  12., -10.,  -6.,  -10., -28.],
                [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,   9.,   
                   6.,  -6.,   6.,  -7.,  -7.,  -6.,   -7., -11.]])
    h = matrix([-3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,  68., 
                 -30., -19., -30.,  99.,  23., -19.,   23.,  10.] )

    A = matrix(0.0, (0, c.size[0]))
    b = matrix(0.0, (0, 1))

    dims = {'l': 2, 'q': [4, 4], 's': [3]}

    #localcones.options.update(opts)
    #sol = localcones.conelp(c, G, h, dims, kktsolver='qr')
    solvers.options.update(opts)
    sol = solvers.conelp(c, G, h, dims)
    print("\nStatus: " + sol['status'])
    if sol['status'] == 'optimal':
        print "x=\n", helpers.strSpe(sol['x'], "%.5f")
        print "s=\n", helpers.strSpe(sol['s'], "%.5f")
        print "z=\n", helpers.strSpe(sol['z'], "%.5f")
        rungotest(sol)

def rungotest(sol):
    print "\n *** running GO test ***"
    helpers.run_go_test("../testconelp", {'x': sol['x'], 's': sol['s'], 'z': sol['z']})


if len(sys.argv[1:]) > 0:
     # if using this use localcones.conelp  instead of solvers.conelp
    if sys.argv[1] == "-sp":
        helpers.sp_reset("./sp.data")
        helpers.sp_activate()

solve({'maxiters': 20})
