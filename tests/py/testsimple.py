
import sys
from cvxopt import matrix, solvers  
import localcones
import helpers

def testsimple(opts):
    c = matrix([0., 1., 0.])
    A = matrix([[1., -1.], [0., 1.], [0., 1.0]])
    b = matrix([1., 0.])
    G = matrix([[0], [-1.], [1.]])
    h = matrix([0.])

    print "c=\n", c
    print "A=\n", A
    print "b=\n", b
    print "G=\n", G
    print "h=\n", h

    #localcones.options.update(opts)
    #sol = localcones.lp(c, G, h, A, b)  
    solvers.options.update(opts)
    sol = solvers.lp(c, G, h, A, b)  
    print"x = \n", helpers.str2(sol['x'], "%.9f")
    print"s = \n", helpers.str2(sol['s'], "%.9f")
    print"z = \n", helpers.str2(sol['z'], "%.9f")
    print "\n *** running GO test ***"
    helpers.run_go_test("../testsimple", {'x': sol['x'], 's': sol['s'], 'z': sol['z']})


if len(sys.argv[1:]) > 0:
    if sys.argv[1] == "-sp":
        helpers.sp_reset("./sp.data")
        helpers.sp_activate()

testsimple({'solver': 'chol2'})

