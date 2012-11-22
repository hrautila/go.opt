
import sys
from cvxopt import matrix, solvers  
import localcones
import helpers

def testlp(opts):
    c = matrix([-4., -5.])  
    G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])  
    h = matrix([3., 3., 0., 0.])  
    #localcones.options.update(opts)
    #sol = localcones.lp(c, G, h)  
    solvers.options.update(opts)
    sol = solvers.lp(c, G, h)  
    print"x = \n", helpers.str2(sol['x'], "%.9f")
    print"s = \n", helpers.str2(sol['s'], "%.9f")
    print"z = \n", helpers.str2(sol['z'], "%.9f")
    print "\n *** running GO test ***"
    helpers.run_go_test("../testlp", {'x': sol['x'], 's': sol['s'], 'z': sol['z']})


if len(sys.argv[1:]) > 0:
    if sys.argv[1] == "-sp":
        helpers.sp_reset("./sp.data")
        helpers.sp_activate()

testlp({'solver': 'chol2'})

