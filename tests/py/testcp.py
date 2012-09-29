# The analytic centering with cone constraints example of section 9.1 
# (Problems with nonlinear objectives).

import sys
from cvxopt import matrix, log, div, spdiag 
#from cvxopt import solvers  
import localcvx, helpers
 
def F(x = None, z = None):  
     if x is None:
          return 0, matrix(0.0, (3,1))  
     if max(abs(x)) >= 1.0:
          return None  
     u = 1 - x**2  
     val = -sum(log(u))  
     Df = div(2*x, u).T  
     if z is None:
          return val, Df  
     H = spdiag(2 * z[0] * div(1 + u**2, u**2))  
     return val, Df, H  
 
def testcp(opts):
     G = matrix([ 
               [0., -1.,  0.,  0., -21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
               [0.,  0., -1.,  0.,   0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
               [0.,  0.,  0., -1.,  -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.]
               ])  
     h = matrix(
          [1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.])  
     dims = {'l': 0, 'q': [4], 's':  [3]}  
     if opts:
          localcvx.options.update(opts)
     sol = localcvx.cp(F, G, h, dims)  
     if sol['status'] == 'optimal':
          print("\nx = \n") 
          print helpers.strSpe(sol['x'], "%.17f")
          print helpers.strSpe(sol['znl'], "%.17f")
          print "\n *** running GO test ***"
          #helpers.run_go_test("../testcp", {'x': sol['x']})


if len(sys.argv[1:]) > 0:
     if sys.argv[1] == '-sp':
          helpers.sp_reset("./sp.cp")
          helpers.sp_activate()

testcp({'maxiters':30})
