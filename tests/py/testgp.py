# The small GP of section 9.3 (Geometric programming).

import sys
from cvxopt import matrix, log, exp, solvers  
import localcvx
import helpers
 
def testgp(opts):
    Aflr  = 1000.0  
    Awall = 100.0  
    alpha = 0.5  
    beta  = 2.0  
    gamma = 0.5  
    delta = 2.0  
 
    F = matrix( [[-1., 1., 1., 0., -1.,  1.,  0.,  0.],  
                 [-1., 1., 0., 1.,  1., -1.,  1., -1.],  
                 [-1., 0., 1., 1.,  0.,  0., -1.,  1.]])  
    g = log( matrix( [1.0, 2/Awall, 2/Awall, 1/Aflr, alpha, 1/beta, gamma, 
                      1/delta]) )  
    K = [1, 2, 1, 1, 1, 1, 1]  
    solvers.options.update(opts)
    sol = solvers.gp(K, F, g)
    #localcvx.options.update(opts)
    #sol = localcvx.gp(K, F, g, kktsolver='chol')
    if sol['status'] == 'optimal':
        x = sol['x']
        print "x=\n", helpers.strSpe(x, "%.17f")
        h, w, d = exp(x)
        print("\n h = %f,  w = %f, d = %f.\n" %(h,w,d))   
        print "\n *** running GO test ***"
        helpers.run_go_test("../testgp", {'x': x})
        

if len(sys.argv[1:]) > 0:
     # if using this use localcvx.cp  instead of solvers.cp
    if sys.argv[1] == '-sp':
        helpers.sp_reset("./sp.data")
        helpers.sp_activate()
        
testgp({'maxiters': 30})

