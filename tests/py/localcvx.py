"""
Convex programming solver.

A primal-dual interior-point solver written in Python and interfaces
for quadratic and geometric programming.  Also includes an interface
to the quadratic programming solver from MOSEK.
"""

# Copyright 2012 M. Andersen and L. Vandenberghe.
# Copyright 2010-2011 L. Vandenberghe.
# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
# 
# This file is part of CVXOPT version 1.1.5.
#
# CVXOPT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# CVXOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import localmisc
import helpers

__all__ = []
options = {}


def cpl(c, F, G = None, h = None, dims = None, A = None, b = None, 
    kktsolver = None, xnewcopy = None, xdot = None, xaxpy = None,
    xscal = None, ynewcopy = None, ydot = None, yaxpy = None, 
    yscal = None):

    """
    Solves a convex optimization problem with a linear objective

        minimize    c'*x 
        subject to  f(x) <= 0
                    G*x <= h
                    A*x = b.                      

    """

    import math 
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix 

    STEP = 0.99
    BETA = 0.5
    ALPHA = 0.01
    EXPON = 3
    MAX_RELAXED_ITERS = 8

    try: DEBUG = options['debug']
    except KeyError: DEBUG = False

    try: MAXITERS = options['maxiters']
    except KeyError: MAXITERS = 100
    else: 
        if type(MAXITERS) is not int or MAXITERS < 1: 
            raise ValueError("options['maxiters'] must be a positive "\
                "integer")

    try: ABSTOL = options['abstol']
    except KeyError: ABSTOL = 1e-7
    else: 
        if type(ABSTOL) is not float and type(ABSTOL) is not int: 
            raise ValueError("options['abstol'] must be a scalar")

    try: RELTOL = options['reltol']
    except KeyError: RELTOL = 1e-6
    else: 
        if type(RELTOL) is not float and type(RELTOL) is not int: 
            raise ValueError("options['reltol'] must be a scalar")

    try: FEASTOL = options['feastol']
    except KeyError: FEASTOL = 1e-7
    else: 
        if (type(FEASTOL) is not float and type(FEASTOL) is not int) or \
            FEASTOL <= 0.0: 
            raise ValueError("options['feastol'] must be a positive "\
                "scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0:
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    try: show_progress = options['show_progress']
    except KeyError: show_progress = True

    try: refinement = options['refinement']
    except KeyError: refinement = 1
    else:
        if type(refinement) is not int or refinement < 0:
            raise ValueError("options['refinement'] must be a "\
                "nonnegative integer")

    if kktsolver is None: 
        if dims and (dims['q'] or dims['s']):  
            kktsolver = 'chol'            
        else:
            kktsolver = 'chol2'            
    defaultsolvers = ('ldl', 'ldl2', 'chol', 'chol2')
    if type(kktsolver) is str and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" \
            %kktsolver)

    try: mnl, x0 = F()   
    except: raise ValueError("function call 'F()' failed")
    
    # Argument error checking depends on level of customization.
    customkkt = type(kktsolver) is not str
    operatorG = G is not None and type(G) not in (matrix, spmatrix)
    operatorA = A is not None and type(A) not in (matrix, spmatrix)
    if (operatorG or operatorA) and not customkkt:
        raise ValueError("use of function valued G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or 
        xscal != None)
    if customx and (not operatorG or not operatorA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or 
        yscal != None) 
    if customy and (not operatorA or not customkkt):
        raise ValueError("use of non vector type for y requires "\
            "function valued A and user-provided kktsolver")

    if not customx:  
        if type(x0) is not matrix or x0.typecode != 'd' or x0.size[1] != 1:
            raise TypeError("'x0' must be a 'd' matrix with one column")
        if type(c) is not matrix or c.typecode != 'd' or c.size != x0.size:
            raise TypeError("'c' must be a 'd' matrix of size (%d,%d)"\
                %(x0.size[0],1))
        
    if h is None: h = matrix(0.0, (0,1))
    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with 1 column")

    if not dims:  dims = {'l': h.size[0], 'q': [], 's': []}

    # Dimension of the product cone of the linear inequalities. with 's' 
    # components unpacked.
    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    if G is None:
        if customx:
            def G(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            G = spmatrix([], [], [], (0, c.size[0]))
    if not operatorG:
        if G.typecode != 'd' or G.size != (cdim, c.size[0]):
            raise TypeError("'G' must be a 'd' matrix with size (%d, %d)"\
                %(cdim, c.size[0]))
        def fG(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha, 
                beta = beta)
    else:
        fG = G

    if A is None:
        if customx or customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: yscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, c.size[0]))
    if not operatorA:
        if A.typecode != 'd' or A.size[1] != c.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns" \
                %c.size[0])
        def fA(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        fA = A
    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if not operatorA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    if b is None and customy:  
        raise ValueEror("use of non vector type for y requires b")

   
    # kktsolver(x, z, W) returns a routine for solving
    #
    #     [ sum_k zk*Hk(x)  A'   GG'*W^{-1} ] [ ux ]   [ bx ]
    #     [ A               0    0          ] [ uy ] = [ by ]
    #     [ GG              0   -W'         ] [ uz ]   [ bz ]
    #
    # where G = [Df(x); G].

    if kktsolver in defaultsolvers:
         if kktsolver == 'ldl': 
             factor = localmisc.kkt_ldl(G, dims, A, mnl)
         elif kktsolver == 'ldl2': 
             factor = misc.kkt_ldl2(G, dims, A, mnl)
         elif kktsolver == 'chol':
             factor = localmisc.kkt_chol(G, dims, A, mnl)
         else: 
             factor = localmisc.kkt_chol2(G, dims, A, mnl)
         def kktsolver(x, z, W):
             f, Df, H = F(x, z)
             return factor(W, H, Df)             


    if xnewcopy is None: xnewcopy = matrix 
    if xdot is None: xdot = blas.dot
    if xaxpy is None: xaxpy = blas.axpy 
    if xscal is None: xscal = blas.scal 
    def xcopy(x, y): 
        xscal(0.0, y) 
        xaxpy(x, y)
    if ynewcopy is None: ynewcopy = matrix 
    if ydot is None: ydot = blas.dot 
    if yaxpy is None: yaxpy = blas.axpy 
    if yscal is None: yscal = blas.scal
    def ycopy(x, y): 
        yscal(0.0, y) 
        yaxpy(x, y)
             

    # Initial points
    x = xnewcopy(x0)
    y = ynewcopy(b);  yscal(0.0, y)
    z, s = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))
    z[: mnl+dims['l']] = 1.0 
    s[: mnl+dims['l']] = 1.0 
    ind = mnl + dims['l']
    for m in dims['q']:
        z[ind] = 1.0
        s[ind] = 1.0
        ind += m
    for m in dims['s']:
        z[ind : ind + m*m : m+1] = 1.0
        s[ind : ind + m*m : m+1] = 1.0
        ind += m**2


    rx, ry = xnewcopy(x0), ynewcopy(b)
    rznl, rzl = matrix(0.0, (mnl, 1)), matrix(0.0, (cdim, 1)), 
    dx, dy = xnewcopy(x), ynewcopy(y)   
    dz, ds = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))

    lmbda = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    lmbdasq = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))

    dz2, ds2 = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))

    newx, newy = xnewcopy(x),  ynewcopy(y)
    newz, news = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))
    newrx = xnewcopy(x0)
    newrznl = matrix(0.0, (mnl, 1))

    rx0, ry0 = xnewcopy(x0), ynewcopy(b)
    rznl0, rzl0 = matrix(0.0, (mnl, 1)), matrix(0.0, (cdim, 1)), 
    x0, dx0 = xnewcopy(x), xnewcopy(dx)
    y0, dy0 = ynewcopy(y), ynewcopy(dy)
    z0 = matrix(0.0, (mnl + cdim, 1))
    dz0 = matrix(0.0, (mnl + cdim, 1))
    dz20 = matrix(0.0, (mnl + cdim, 1))
    s0 = matrix(0.0, (mnl + cdim, 1))
    ds0 = matrix(0.0, (mnl + cdim, 1))
    ds20 = matrix(0.0, (mnl + cdim, 1))
    W0 = {}
    W0['dnl'] = matrix(0.0, (mnl, 1))
    W0['dnli'] = matrix(0.0, (mnl, 1))
    W0['d'] = matrix(0.0, (dims['l'], 1))
    W0['di'] = matrix(0.0, (dims['l'], 1))
    W0['v'] = [ matrix(0.0, (m, 1)) for m in dims['q'] ]
    W0['beta'] = len(dims['q']) * [ 0.0 ]
    W0['r'] = [ matrix(0.0, (m, m)) for m in dims['s'] ]
    W0['rti'] = [ matrix(0.0, (m, m)) for m in dims['s'] ]
    lmbda0 = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    lmbdasq0 = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    

    if show_progress: 
        print("% 10s% 12s% 10s% 8s% 7s" %("pcost", "dcost", "gap", "pres",
            "dres"))

    #print "x", type(x)
    #print "y", type(y)
    #print "s", type(s)
    #print "z", type(z)
    #print "rx", type(rx)
    #print "ry", type(ry)
    #print "rzl", type(rzl)
    #print "rznl", type(rznl)
    #print "x0", type(x0)
    #print "y0", type(y0)
    #print "s0", type(s0)
    #print "z0", type(z0)

    helpers.sp_add_var("c", c)
    helpers.sp_add_var("b", b)
    helpers.sp_add_var("x", x)
    helpers.sp_add_var("y", y)
    helpers.sp_add_var("z", z)
    helpers.sp_add_var("s", s)
    helpers.sp_add_var("dx", dx)
    helpers.sp_add_var("dy", dy)
    helpers.sp_add_var("dz", dz)
    helpers.sp_add_var("ds", ds)
    helpers.sp_add_var("x0", x0)
    helpers.sp_add_var("rx", rx)
    helpers.sp_add_var("rznl", rznl)
    helpers.sp_add_var("rzl", rzl)
    helpers.sp_add_var("lmbda", lmbda)
    helpers.sp_add_var("lmbdasq", lmbdasq)
    
    #print "preloop c=\n", helpers.str2(c, "%.7f")
    relaxed_iters = 0
    for iters in range(MAXITERS + 1):  
        helpers.sp_major_next()
        helpers.sp_create("loopstart", 10)

        if refinement or DEBUG:  
            # We need H to compute residuals of KKT equations.
            f, Df, H = F(x, z[:mnl])
        else:
            #print "%d get f, Df, ..." % iters
            f, Df = F(x)
       
        if iters == 0:
            #print "f", type(f)
            #print "Df", type(Df)
            #print "H", type(H)
            pass

        f = matrix(f, tc='d')
        if f.typecode != 'd' or f.size != (mnl, 1):
            raise TypeError("first output argument of F() must be a "\
                "'d' matrix of size (%d, %d)" %(mnl, 1))

        if type(Df) is matrix or type(Df) is spmatrix:
            if customx: raise ValueError("use of non-vector type for x "\
                "requires function valued Df")
            if Df.typecode != 'd' or Df.size != (mnl, c.size[0]):
                raise TypeError("second output argument of F() must "\
                    "be a 'd' matrix of size (%d,%d)" %(mnl, c.size[0]))
            def fDf(u, v, alpha = 1.0, beta = 0.0, trans = 'N'): 
                base.gemv(Df, u, v, alpha = alpha, beta = beta, trans = 
                    trans)
        else: 
            if not customkkt:
                raise ValueError("use of function valued Df requires "\
                    "a user-provided kktsolver")
            fDf = Df

        if refinement or DEBUG:
            if type(H) is matrix or type(H) is spmatrix:
                if customx: raise ValueError("use of non-vector type "\
                    "for  x requires function valued H")
                if H.typecode != 'd' or H.size != (c.size[0], c.size[0]):
                    raise TypeError("third output argument of F() must "\
                        "be a 'd' matrix of size (%d,%d)" \
                        %(c.size[0], c.size[0]))
                def fH(u, v, alpha = 1.0, beta = 0.0): 
                    base.symv(H, u, v, alpha = alpha, beta = beta)
            else: 
                if not customkkt:
                    raise ValueError("use of function valued H requires "\
                        "a user-provided kktsolver")
                fH = H
           

        gap = misc.sdot(s, z, dims, mnl) 
        #print "%d: gap = %.9f" % (iters, gap)

        # rx = c + A'*y + Df'*z[:mnl] + G'*z[mnl:]
        xcopy(c, rx) 
        fA(y, rx, beta = 1.0, trans = 'T')
        fDf(z[:mnl], rx, beta = 1.0, trans = 'T')
        fG(z[mnl:], rx, beta = 1.0, trans = 'T')
        #print "3 rx=\n", helpers.str2(rx, "%.7f")
        resx = math.sqrt(xdot(rx, rx))
           
        # ry = A*x - b
        ycopy(b, ry)
        fA(x, ry, alpha = 1.0, beta = -1.0)
        resy = math.sqrt(ydot(ry, ry))

        # rznl = s[:mnl] + f 
        blas.copy(s[:mnl], rznl)
        blas.axpy(f, rznl)
        #print "rznl=\n", helpers.str2(rznl, "%.7f")
        #print "f=\n", helpers.str2(f, "%.7f")
        resznl = blas.nrm2(rznl)

        # rzl = s[mnl:] + G*x - h
        blas.copy(s[mnl:], rzl)
        blas.axpy(h, rzl, alpha = -1.0)
        fG(x, rzl, beta = 1.0)
        reszl = misc.snrm2(rzl, dims)
        #print "%d: resx = %.9f, resznl = %.9f reszl = %.9f" % (iters, resx, resznl, reszl)

        # Statistics for stopping criteria.

        # pcost = c'*x
        # dcost = c'*x + y'*(A*x-b) + znl'*f(x) + zl'*(G*x-h)
        #       = c'*x + y'*(A*x-b) + znl'*(f(x)+snl) + zl'*(G*x-h+sl) 
        #         - z'*s
        #       = c'*x + y'*ry + znl'*rznl + zl'*rzl - gap
        pcost = xdot(c,x)
        dcost = pcost + ydot(y, ry) + blas.dot(z[:mnl], rznl) + \
            misc.sdot(z[mnl:], rzl, dims) - gap
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None
        pres = math.sqrt( resy**2 + resznl**2 + reszl**2 )
        dres = resx
        if iters == 0: 
            resx0 = max(1.0, resx)
            resznl0 = max(1.0, resznl)
            pres0 = max(1.0, pres)
            dres0 = max(1.0, dres)
            gap0 = gap
            theta1 = 1.0 / gap0
            theta2 = 1.0 / resx0
            theta3 = 1.0 / resznl0
        phi = theta1 * gap + theta2 * resx + theta3 * resznl
        pres = pres / pres0
        dres = dres / dres0

        if show_progress:
            print("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e" \
                %(iters, pcost, dcost, gap, pres, dres))

        helpers.sp_create("checkgap", 50, {"gap": gap,
                                           "pcost": pcost,
                                           "dcost": dcost,
                                           "pres": pres,
                                           "dres": dres,
                                           "resx": resx,
                                           "resy": resy,
                                           "reszl": reszl,
                                           "resznl": resznl
                                           })
        # Stopping criteria.    
        if ( pres <= FEASTOL and dres <= FEASTOL and ( gap <= ABSTOL or 
            (relgap is not None and relgap <= RELTOL) )) or \
            iters == MAXITERS:
            sl, zl = s[mnl:], z[mnl:]
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(sl, m, ind)
                misc.symm(zl, m, ind)
                ind += m**2
            ts = misc.max_step(s, dims, mnl)
            tz = misc.max_step(z, dims, mnl)
            if iters == MAXITERS:
                if show_progress:
                    print("Terminated (maximum number of iterations "\
                        "reached).")
                status = 'unknown'
            else:
                if show_progress:
                    print("Optimal solution found.")
                status = 'optimal'

            return {'status': status, 'x': x,  'y': y, 'znl': z[:mnl],  
                'zl': zl, 'snl': s[:mnl], 'sl': sl, 'gap': gap, 
                'relative gap': relgap, 'primal objective': pcost, 
                'dual objective': dcost,  'primal slack': -ts, 
                'dual slack': -tz, 'primal infeasibility': pres,
                'dual infeasibility': dres }


        # Compute initial scaling W: 
        #
        #     W * z = W^{-T} * s = lambda.
        #
        # lmbdasq = lambda o lambda 

        if iters == 0:  
            W = misc.compute_scaling(s, z, lmbda, dims, mnl)
            helpers.sp_add_var("W", W)
        misc.ssqr(lmbdasq, lmbda, dims, mnl)
        #print "lmbdasq=\n", helpers.str2(lmbda, "%.9f")
        helpers.sp_create("lmbdasq", 90)

        # f3(x, y, z) solves
        #
        #     [ H   A'  GG'*W^{-1} ] [ ux ]   [ bx ]
        #     [ A   0   0          ] [ uy ] = [ by ].
        #     [ GG  0  -W'         ] [ uz ]   [ bz ]
        #
        # On entry, x, y, z contain bx, by, bz.
        # On exit, they contain ux, uy, uz.
        
        try:
            helpers.sp_minor_push(95)
            f3 = kktsolver(x, z[:mnl], W)
            helpers.sp_minor_pop()
            helpers.sp_create("f3", 100)
            #print "x=\n", helpers.str2(x,"%.7f")
            #print "z=\n", helpers.str2(z,"%.7f")
        except ArithmeticError: 
            singular_kkt_matrix = False
            if iters == 0:
                raise ValueError("Rank(A) < p or "\
                    "Rank([H(x); A; Df(x); G]) < n")

            elif 0 < relaxed_iters < MAX_RELAXED_ITERS > 0:
                # The arithmetic error may be caused by a relaxed line 
                # search in the previous iteration.  Therefore we restore 
                # the last saved state and require a standard line search. 

                print "via ArithmeticError ..."
                phi, gap = phi0, gap0
                mu = gap / ( mnl + dims['l'] + len(dims['q']) + 
                    sum(dims['s']) )
                blas.copy(W0['dnl'], W['dnl'])
                blas.copy(W0['dnli'], W['dnli'])
                blas.copy(W0['d'], W['d'])
                blas.copy(W0['di'], W['di'])
                for k in range(len(dims['q'])):
                    blas.copy(W0['v'][k], W['v'][k])
                    W['beta'][k] = W0['beta'][k]
                for k in range(len(dims['s'])):
                    blas.copy(W0['r'][k], W['r'][k])
                    blas.copy(W0['rti'][k], W['rti'][k])
                xcopy(x0, x); 
                ycopy(y0, y); 
                blas.copy(s0, s); blas.copy(z0, z)
                blas.copy(lmbda0, lmbda)
                blas.copy(lmbdasq, lmbdasq0)
                xcopy(rx0, rx); ycopy(ry0, ry)
                resx = math.sqrt(xdot(rx, rx))
                blas.copy(rznl0, rznl);  blas.copy(rzl0, rzl);
                resznl = blas.nrm2(rznl)

                relaxed_iters = -1

                try:
                    helpers.cp_minor_push(120)
                    f3 = kktsolver(x, z[:mnl], W)
                    helpers.cp_minor_pop()
                except ArithmeticError: 
                     singular_kkt_matrix = True

            else:  
                 singular_kkt_matrix = True

            if singular_kkt_matrix:
                sl, zl = s[mnl:], z[mnl:]
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(sl, m, ind)
                    misc.symm(zl, m, ind)
                    ind += m**2
                ts = misc.max_step(s, dims, mnl)
                tz = misc.max_step(z, dims, mnl)
                if show_progress:
                    print("Terminated (singular KKT matrix).")
                status = 'unknown'
                return {'status': status, 'x': x,  'y': y, 
                    'znl': z[:mnl],  'zl': zl, 'snl': s[:mnl], 
                    'sl': sl, 'gap': gap, 'relative gap': relgap, 
                    'primal objective': pcost, 'dual objective': dcost,  
                    'primal infeasibility': pres, 
                    'dual infeasibility': dres, 'primal slack': -ts,
                    'dual slack': -tz }


        # f4_no_ir(x, y, z, s) solves
        # 
        #     [ 0     ]   [ H   A'  GG' ] [ ux        ]   [ bx ]
        #     [ 0     ] + [ A   0   0   ] [ uy        ] = [ by ]
        #     [ W'*us ]   [ GG  0   0   ] [ W^{-1}*uz ]   [ bz ]
        #
        #     lmbda o (uz + us) = bs.
        #
        # On entry, x, y, z, x, contain bx, by, bz, bs.
        # On exit, they contain ux, uy, uz, us.

        #print "dx=\n", helpers.str2(dx,"%.7f")
        #print "dz=\n", helpers.str2(dz,"%.7f")

        if iters == 0:
            ws3 = matrix(0.0, (mnl + cdim, 1))
            wz3 = matrix(0.0, (mnl + cdim, 1))
            helpers.sp_add_var("ws3", ws3)
            helpers.sp_add_var("wz3", wz3)

        def f4_no_ir(x, y, z, s):

            # Solve 
            #
            #     [ H  A'  GG'  ] [ ux        ]   [ bx                    ]
            #     [ A  0   0    ] [ uy        ] = [ by                    ]
            #     [ GG 0  -W'*W ] [ W^{-1}*uz ]   [ bz - W'*(lmbda o\ bs) ]
            #
            #     us = lmbda o\ bs - uz.
            
            # s := lmbda o\ s 
            #    = lmbda o\ bs
            misc.sinv(s, lmbda, dims, mnl)

            # z := z - W'*s 
            #    = bz - W' * (lambda o\ bs)
            blas.copy(s, ws3)
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, z, alpha = -1.0)

            # Solve for ux, uy, uz
            f3(x, y, z)

            # s := s - z 
            #    = lambda o\ bs - z.
            blas.axpy(z, s, alpha = -1.0)


        if iters == 0:
            wz2nl, wz2l = matrix(0.0, (mnl,1)), matrix(0.0, (cdim, 1))
            helpers.sp_add_var("wz2nl", wz2nl)
            helpers.sp_add_var("wz2l", wz2l)

        def res(ux, uy, uz, us, vx, vy, vz, vs):

            # Evaluates residuals in Newton equations:
            #
            #     [ vx ]     [ 0     ]   [ H  A' GG' ] [ ux        ]
            #     [ vy ] -=  [ 0     ] + [ A  0  0   ] [ uy        ]
            #     [ vz ]     [ W'*us ]   [ GG 0  0   ] [ W^{-1}*uz ]
            #
            #     vs -= lmbda o (uz + us).
            minor = helpers.sp_minor_top()
            # vx := vx - H*ux - A'*uy - GG'*W^{-1}*uz
            fH(ux, vx, alpha = -1.0, beta = 1.0)
            helpers.sp_create("00res", minor+10)
            fA(uy, vx, alpha = -1.0, beta = 1.0, trans = 'T') 
            blas.copy(uz, wz3)
            helpers.sp_create("02res", minor+10)
            misc.scale(wz3, W, inverse = 'I')
            fDf(wz3[:mnl], vx, alpha = -1.0, beta = 1.0, trans = 'T')
            helpers.sp_create("04res", minor+10)
            fG(wz3[mnl:], vx, alpha = -1.0, beta = 1.0, trans = 'T') 

            helpers.sp_create("10res", minor+10)

            # vy := vy - A*ux 
            fA(ux, vy, alpha = -1.0, beta = 1.0)

            # vz := vz - W'*us - GG*ux 
            fDf(ux, wz2nl)
            helpers.sp_create("15res", minor+10)
            blas.axpy(wz2nl, vz, alpha = -1.0)
            fG(ux, wz2l)
            helpers.sp_create("20res", minor+10)
            blas.axpy(wz2l, vz, alpha = -1.0, offsety = mnl)
            blas.copy(us, ws3) 
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, vz, alpha = -1.0)
            helpers.sp_create("30res", minor+10)

            # vs -= lmbda o (uz + us)
            blas.copy(us, ws3)
            blas.axpy(uz, ws3)
            misc.sprod(ws3, lmbda, dims, mnl, diag = 'D')
            blas.axpy(ws3, vs, alpha = -1.0)
            helpers.sp_create("90res", minor+10)


        # f4(x, y, z, s) solves the same system as f4_no_ir, but applies
        # iterative refinement.

        if iters == 0:
            if refinement or DEBUG:
                wx, wy = xnewcopy(c), ynewcopy(b)
                wz = matrix(0.0, (mnl + cdim, 1))
                ws = matrix(0.0, (mnl + cdim, 1))
                helpers.sp_add_var("wx", wx)
                helpers.sp_add_var("wz", wz)
                helpers.sp_add_var("ws", ws)
            if refinement:
                wx2, wy2 = xnewcopy(c), ynewcopy(b)
                wz2 = matrix(0.0, (mnl + cdim, 1))
                ws2 = matrix(0.0, (mnl + cdim, 1))
                helpers.sp_add_var("wx2", wx2)
                helpers.sp_add_var("wz2", wz2)
                helpers.sp_add_var("ws2", ws2)

        def f4(x, y, z, s):
            minor = helpers.sp_minor_top()
            if refinement or DEBUG: 
                xcopy(x, wx)        
                ycopy(y, wy)        
                blas.copy(z, wz)        
                blas.copy(s, ws)        
            #print "--- pre f4_no_ir"
            #print "x=\n", helpers.str2(x,"%.7f")
            #print "z=\n", helpers.str2(z,"%.7f")
            #print "s=\n", helpers.str2(s,"%.7f")
            #print "--- end of pre f4_no_ir"
            helpers.sp_create("0_f4", minor+100)
            helpers.sp_minor_push(minor+100)
            f4_no_ir(x, y, z, s)        
            helpers.sp_minor_pop()
            helpers.sp_create("1_f4", minor+200)
            #print "--- post f4_no_ir"
            #print "x=\n", helpers.str2(x,"%.7f")
            #print "z=\n", helpers.str2(z,"%.7f")
            #print "s=\n", helpers.str2(s,"%.7f")
            #print "--- end of post f4_no_ir"
            for i in range(refinement):
                xcopy(wx, wx2)        
                ycopy(wy, wy2)        
                blas.copy(wz, wz2)        
                blas.copy(ws, ws2)        
                helpers.sp_create("2_f4", minor+(1+i)*200)
                helpers.sp_minor_push(minor+(1+i)*200)
                #print "-- pre-res wx2=\n", helpers.str2(wx2,"%.7f")
                res(x, y, z, s, wx2, wy2, wz2, ws2) 
                helpers.sp_minor_pop()
                helpers.sp_create("3_f4", minor+(1+i)*200+100)
                helpers.sp_minor_push(minor+(1+i)*200+100)
                f4_no_ir(wx2, wy2, wz2, ws2)
                helpers.sp_minor_pop()
                helpers.sp_create("4_f4", minor+(1+i)*200+199)
                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                blas.axpy(ws2, s)
            if DEBUG:
                res(x, y, z, s, wx, wy, wz, ws)
                print("KKT residuals:")
                print("    'x': %e" %math.sqrt(xdot(wx, wx)))
                print("    'y': %e" %math.sqrt(ydot(wy, wy)))
                print("    'z': %e" %misc.snrm2(wz, dims, mnl))
                print("    's': %e" %misc.snrm2(ws, dims, mnl))
     

        sigma, eta = 0.0, 0.0
        #print "pre loop [0,1]"
        #print "x=\n", helpers.str2(x,"%.7f")
        #print "z=\n", helpers.str2(z,"%.7f")
        #print "s=\n", helpers.str2(s,"%.7f")
        #print "dx=\n", helpers.str2(dx,"%.7f")
        #print "dz=\n", helpers.str2(dz,"%.7f")
        #print "ds=\n", helpers.str2(ds,"%.7f")
        #print "rx=\n", helpers.str2(rx,"%.7f")
        #print "rzl=\n", helpers.str2(rzl,"%.7f")
        #print "rznl=\n", helpers.str2(rznl,"%.7f")
        #print "lmbda=\n", helpers.str2(lmbda,"%.7f")
        #print "lmbdasq=\n", helpers.str2(lmbdasq,"%.7f")
        #helpers.printW(W)

        for i in [0, 1]:
            minor = (i+2)*1000
            helpers.sp_minor_push(minor)
            helpers.sp_create("loop01", minor)
            #print "beginning loop [0,1]"

            # Solve
            #
            #     [ 0     ]   [ H  A' GG' ] [ dx        ]
            #     [ 0     ] + [ A  0  0   ] [ dy        ] = -(1 - eta)*r  
            #     [ W'*ds ]   [ GG 0  0   ] [ W^{-1}*dz ]
            #
            #     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e.
            #

            mu = gap / (mnl + dims['l'] + len(dims['q']) + sum(dims['s']))

            # ds = -lmbdasq + sigma * mu * e  
            blas.scal(0.0, ds)
            blas.axpy(lmbdasq, ds, n = mnl + dims['l'] + sum(dims['q']), 
                alpha = -1.0)
            ds[:mnl + dims['l']] += sigma*mu
            ind = mnl + dims['l']
            for m in dims['q']:
                ds[ind] += sigma*mu
                ind += m
            ind2 = ind
            for m in dims['s']:
                blas.axpy(lmbdasq, ds, n = m, offsetx = ind2, offsety =  
                    ind, incy = m + 1, alpha = -1.0)
                ds[ind : ind + m*m : m+1] += sigma*mu
                ind += m*m
                ind2 += m
       
            # (dx, dy, dz) := -(1-eta) * (rx, ry, rz)
            xscal(0.0, dx);  xaxpy(rx, dx, alpha = -1.0 + eta)
            yscal(0.0, dy);  yaxpy(ry, dy, alpha = -1.0 + eta)
            blas.scal(0.0, dz) 
            blas.axpy(rznl, dz, alpha = -1.0 + eta)
            blas.axpy(rzl, dz, alpha = -1.0 + eta, offsety = mnl)

            #print "dx=\n", helpers.str2(dx,"%.7f")
            #print "dz=\n", helpers.str2(dz,"%.7f")
            #print "ds=\n", helpers.str2(ds,"%.7f")
            
            helpers.sp_create("pref4", minor)
            helpers.sp_minor_push(minor)

            try:
                f4(dx, dy, dz, ds)
            except ArithmeticError: 
                if iters == 0:
                    raise ValueError("Rank(A) < p or "\
                        "Rank([H(x); A; Df(x); G]) < n")
                else:
                    sl, zl = s[mnl:], z[mnl:]
                    ind = dims['l'] + sum(dims['q'])
                    for m in dims['s']:
                        misc.symm(sl, m, ind)
                        misc.symm(zl, m, ind)
                        ind += m**2
                    ts = misc.max_step(s, dims, mnl)
                    tz = misc.max_step(z, dims, mnl)
                    if show_progress:
                        print("Terminated (singular KKT matrix).")
                    return {'status': 'unknown', 'x': x,  'y': y, 
                        'znl': z[:mnl],  'zl': zl, 'snl': s[:mnl], 
                        'sl': sl, 'gap': gap, 'relative gap': relgap, 
                        'primal objective': pcost, 'dual objective': dcost,
                        'primal infeasibility': pres, 
                        'dual infeasibility': dres, 'primal slack': -ts,
                        'dual slack': -tz }

            #print "dx=\n", helpers.str2(dx,"%.7f")
            #print "dz=\n", helpers.str2(dz,"%.7f")

            helpers.sp_minor_pop()
            helpers.sp_create("postf4", minor+400)

            # Inner product ds'*dz and unscaled steps are needed in the 
            # line search.
            dsdz = misc.sdot(ds, dz, dims, mnl)
            blas.copy(dz, dz2)
            misc.scale(dz2, W, inverse = 'I')
            blas.copy(ds, ds2)
            misc.scale(ds2, W, trans = 'T')

            helpers.sp_create("dsdz", minor+410)

            # Maximum steps to boundary. 
            # 
            # Also compute the eigenvalue decomposition of 's' blocks in 
            # ds, dz.  The eigenvectors Qs, Qz are stored in ds, dz.
            # The eigenvalues are stored in sigs, sigz.

            misc.scale2(lmbda, ds, dims, mnl)
            ts = misc.max_step(ds, dims, mnl, sigs)
            misc.scale2(lmbda, dz, dims, mnl)
            tz = misc.max_step(dz, dims, mnl, sigz)
            t = max([ 0.0, ts, tz ])
            if t == 0:
                step = 1.0
            else:
                step = min(1.0, STEP / t)

            helpers.sp_create("maxstep", minor+420)
            #print "%d: ts=%.7f, tz=%.7f, t=%.7f, step=%.7f" %(iters, ts, tz, t, step)

            # Backtrack until newx is in domain of f.
            backtrack = True
            while backtrack:
                xcopy(x, newx);  xaxpy(dx, newx, alpha = step)
                t = F(newx)
                if t is None: newf = None
                else: newf, newDf = t[0], t[1]
                if newf is not None:
                    backtrack = False
                else:
                    step *= BETA


            # Merit function 
            #
            #     phi = theta1 * gap + theta2 * norm(rx) + 
            #         theta3 * norm(rznl)
            #
            # and its directional derivative dphi.

            phi = theta1 * gap + theta2 * resx + theta3 * resznl
            if i == 0:
                dphi = -phi 
            else:
                dphi = -theta1 * (1 - sigma) * gap -  \
                    theta2 * (1 - eta) * resx - \
                    theta3 * (1 - eta) * resznl  


            # Line search.
            #
            # We use two types of line search.  In a standard iteration we
            # use is a normal backtracking line search terminating with a 
            # sufficient decrease of phi.  In a "relaxed" iteration the 
            # line search is terminated after one step, regardless of the 
            # value of phi.  We make at most MAX_RELAXED_ITERS consecutive
            # relaxed iterations.  When starting a series of relaxed 
            # iteration, we save the state at the end of the first line 
            # search in the series (scaling, primal and dual iterates, 
            # steps, sigma, eta, i.e., all information needed to resume 
            # the line search at some later point).  If a series of
            # MAX_RELAXED_ITERS relaxed iterations does not result in a 
            # sufficient decrease compared to the value of phi at the start
            # of the series, then we resume the first line search in the 
            # series as a standard line search (using the saved state).
            # On the other hand, if at some point during the series of 
            # relaxed iterations we obtain a sufficient decrease of phi 
            # compared with the value at the start of the series, then we 
            # start a new series of relaxed line searches. 
            #  
            # To implement this we use a counter relaxed_iters.
            #
            # 1. If 0 <= relaxed_iters < MAX_RELAXED_ITERS, we use a 
            #    relaxed line search (one full step).  If relaxed_iters 
            #    is 0, we save the value phi0 of the merit function at the
            #    current point, as well as the state at the end of the 
            #    line search.  If the relaxed line search results in a 
            #    sufficient decrease w.r.t. phi0, we reset relaxed_iters 
            #    to 0.  Otherwise we increment relaxed_iters.
            #
            # 2. If relaxed_iters is MAX_RELAXED_ITERS, we use a standard
            #    line search.  If this results in a sufficient decrease 
            #    of the merit function compared with phi0, we set 
            #    relaxed_iters to 0.  If phi decreased compared with phi0,
            #    but not sufficiently, we set relaxed_iters to -1.  
            #    If phi increased compared with phi0, we resume the 
            #    backtracking at the last saved iteration, and after 
            #    completing the line search, set relaxed_iters to 0.
            # 
            # 3. If relaxed_iters is -1, we use a standard line search
            #    and increment relaxed_iters to 0. 


            backtrack = True
            while backtrack:
                xcopy(x, newx);  xaxpy(dx, newx, alpha = step)
                ycopy(y, newy);  yaxpy(dy, newy, alpha = step)
                blas.copy(z, newz);  blas.axpy(dz2, newz, alpha = step) 
                blas.copy(s, news);  blas.axpy(ds2, news, alpha = step) 

                t = F(newx)
                newf, newDf = matrix(t[0], tc = 'd'), t[1]
                if type(newDf) is matrix or type(Df) is spmatrix:
                    if newDf.typecode != 'd' or \
                        newDf.size != (mnl, c.size[0]):
                            raise TypeError("second output argument "\
                                "of F() must be a 'd' matrix of size "\
                                "(%d,%d)" %(mnl, c.size[0]))
                    def newfDf(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
                        base.gemv(newDf, u, v, alpha = alpha, beta = 
                                beta, trans = trans)
                else: 
                    newfDf = newDf

                #print "news=\n", helpers.str2(news, "%.7f")
                #print "newf=\n", helpers.str2(newf, "%.7f")

                # newrx = c + A'*newy + newDf'*newz[:mnl] + G'*newz[mnl:]
                xcopy(c, newrx) 
                fA(newy, newrx, beta = 1.0, trans = 'T')
                newfDf(newz[:mnl], newrx, beta = 1.0, trans = 'T')
                fG(newz[mnl:], newrx, beta = 1.0, trans = 'T')
                newresx = math.sqrt(xdot(newrx, newrx))
           
                # newrznl = news[:mnl] + newf 
                blas.copy(news[:mnl], newrznl)
                blas.axpy(newf, newrznl)
                newresznl = blas.nrm2(newrznl)

                newgap = (1.0 - (1.0 - sigma) * step) * gap \
                    + step**2 * dsdz
                newphi = theta1 * newgap  + theta2 * newresx + \
                    theta3 * newresznl

                #print "theta1=%.7f, theta2=%.7f theta3=%.7f" % (theta1, theta2, theta3)
                #print "newgap=%.7f, newphi=%.7f newresx=%.7f newresznl=%.7f" % (newgap, newphi, newresx, newresznl)

                if i == 0:
                    if newgap <= (1.0 - ALPHA * step) * gap and \
                        ( 0 <= relaxed_iters < MAX_RELAXED_ITERS or \
                        newphi <= phi + ALPHA * step * dphi ):
                        backtrack = False
                        sigma = min(newgap/gap, (newgap/gap) ** EXPON)
                        #print "break 1: sigma=%.7f" % sigma
                        eta = 0.0 
                    else:
                        step *= BETA

                else:
                    if relaxed_iters == -1 or ( relaxed_iters == 0 == 
                        MAX_RELAXED_ITERS ):
                        # Do a standard line search.
                        if newphi <= phi + ALPHA * step * dphi:
                            relaxed_iters == 0
                            backtrack = False
                            #print "break 2: newphi=%.7f" % newphi
                        else:
                            step *= BETA

                    elif relaxed_iters == 0 < MAX_RELAXED_ITERS:
                        if newphi <= phi + ALPHA * step * dphi:
                            # Relaxed l.s. gives sufficient decrease.
                            relaxed_iters = 0

                        else:
                            # Save state.
                            phi0, dphi0, gap0 = phi, dphi, gap
                            step0 = step
                            blas.copy(W['dnl'], W0['dnl'])
                            blas.copy(W['dnli'], W0['dnli'])
                            blas.copy(W['d'], W0['d'])
                            blas.copy(W['di'], W0['di'])
                            for k in range(len(dims['q'])):
                                blas.copy(W['v'][k], W0['v'][k])
                                W0['beta'][k] = W['beta'][k]
                            for k in range(len(dims['s'])):
                                blas.copy(W['r'][k], W0['r'][k])
                                blas.copy(W['rti'][k], W0['rti'][k])
                            xcopy(x, x0); xcopy(dx, dx0)
                            ycopy(y, y0); ycopy(dy, dy0)
                            blas.copy(s, s0); blas.copy(z, z0)
                            blas.copy(ds, ds0)
                            blas.copy(dz, dz0)
                            blas.copy(ds2, ds20)
                            blas.copy(dz2, dz20)
                            blas.copy(lmbda, lmbda0)
                            blas.copy(lmbdasq, lmbdasq0)
                            dsdz0 = dsdz
                            sigma0, eta0 = sigma, eta
                            xcopy(rx, rx0);  ycopy(ry, ry0)
                            blas.copy(rznl, rznl0); blas.copy(rzl, rzl0)
                            relaxed_iters = 1

                        backtrack = False
                        #print "break 3: newphi=%.7f" % newphi

                    elif 0 <= relaxed_iters < MAX_RELAXED_ITERS > 0:
                        if newphi <= phi0 + ALPHA * step0 * dphi0:
                            # Relaxed l.s. gives sufficient decrease.
                            relaxed_iters = 0

                        else: 
                            # Relaxed line search 
                            relaxed_iters += 1

                        backtrack = False
                        #print "break 4: newphi=%.7f" % newphi

                    elif relaxed_iters == MAX_RELAXED_ITERS > 0:
                        if newphi <= phi0 + ALPHA * step0 * dphi0:
                            # Series of relaxed line searches ends 
                            # with sufficient decrease w.r.t. phi0.
                            backtrack = False
                            relaxed_iters = 0
                            #print "break 5: newphi=%.7f" % newphi

                        elif newphi >= phi0:
                            # Resume last saved line search.
                            phi, dphi, gap = phi0, dphi0, gap0
                            step = step0
                            blas.copy(W0['dnl'], W['dnl'])
                            blas.copy(W0['dnli'], W['dnli'])
                            blas.copy(W0['d'], W['d'])
                            blas.copy(W0['di'], W['di'])
                            for k in range(len(dims['q'])):
                                blas.copy(W0['v'][k], W['v'][k])
                                W['beta'][k] = W0['beta'][k]
                            for k in range(len(dims['s'])):
                                blas.copy(W0['r'][k], W['r'][k])
                                blas.copy(W0['rti'][k], W['rti'][k])
                            xcopy(x0, x); xcopy(dx0, dx);
                            ycopy(y0, y); ycopy(dy0, dy);
                            blas.copy(s0, s); blas.copy(z0, z)
                            blas.copy(ds0, ds)
                            blas.copy(dz0, dz)
                            blas.copy(ds20, ds2)
                            blas.copy(dz20, dz2)
                            blas.copy(lmbda0, lmbda)
                            dsdz = dsdz0
                            sigma, eta = sigma0, eta0
                            relaxed_iters = -1

                        elif newphi <= phi + ALPHA * step * dphi:
                            # Series of relaxed line searches ends with
                            # with insufficient decrease w.r.t. phi0" 
                            backtrack = False
                            relaxed_iters = -1
                            #print "break 6: newphi=%.7f" % newphi
            
            helpers.sp_create("eol", minor+900)
            #print "eol ds=\n", helpers.str2(ds,"%.7f")
            #print "eol dz=\n", helpers.str2(dz,"%.7f")


        # Update x, y.
        xaxpy(dx, x, alpha = step)
        yaxpy(dy, y, alpha = step)
        #print "update x=\n", helpers.str2(x,"%.7f")
        helpers.sp_create("updatexy", 5000)

        # Replace nonlinear, 'l' and 'q' blocks of ds and dz with the 
        # updated variables in the current scaling.
        # Replace 's' blocks of ds and dz with the factors Ls, Lz in a
        # factorization Ls*Ls', Lz*Lz' of the updated variables in the 
        # current scaling.

        # ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
        # dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
        #print "pre-update ds=\n", helpers.str2(ds,"%.7f")
        #print "pre-update dz=\n", helpers.str2(dz,"%.7f")
        blas.scal(step, ds, n = mnl + dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n = mnl + dims['l'] + sum(dims['q']))
        ind = mnl + dims['l']
        ds[:ind] += 1.0
        dz[:ind] += 1.0
        for m in dims['q']:
            ds[ind] += 1.0
            dz[ind] += 1.0
            ind += m
        #print "update ds=\n", helpers.str2(ds,"%.7f")
        #print "update dz=\n", helpers.str2(dz,"%.7f")
        helpers.sp_create("updatedsdz", 5100)


        # ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        # 
        # This replaces the nonlinear, 'l' and 'q' components of ds and dz
        # with the updated variables in the new scaling.
        # The 's' components of ds and dz are replaced with
        #
        #     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        #     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
         
        misc.scale2(lmbda, ds, dims, mnl, inverse = 'I')
        misc.scale2(lmbda, dz, dims, mnl, inverse = 'I')

        helpers.sp_create("scale2", 5200)
        # sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        # sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n = sum(dims['s']), k = 0, ldA = 1, 
            offsetA = mnl + dims['l'] + sum(dims['q']) )
        blas.tbsv(lmbda, sigz, n = sum(dims['s']), k = 0, ldA = 1, 
            offsetA = mnl + dims['l'] + sum(dims['q']) )

        helpers.sp_create("sigs", 5300)

        # dsk := Ls = dsk * sqrt(sigs).
        # dzk := Lz = dzk * sqrt(sigz).
        ind2, ind3 = mnl + dims['l'] + sum(dims['q']), 0
        for k in range(len(dims['s'])):
            m = dims['s'][k]
            for i in range(m):
                a = math.sqrt(sigs[ind3+i])
                blas.scal(a, ds, offset = ind2 + m*i, n = m)
                a = math.sqrt(sigz[ind3+i])
                blas.scal(a, dz, offset = ind2 + m*i, n = m)
            ind2 += m*m
            ind3 += m


        # Update lambda and scaling.

        helpers.sp_create("scaling", 5400)
        misc.update_scaling(W, lmbda, ds, dz)
        helpers.sp_create("postscaling", 5500)


        # Unscale s, z (unscaled variables are used only to compute 
        # feasibility residuals).

        blas.copy(lmbda, s, n = mnl + dims['l'] + sum(dims['q']))
        ind = mnl + dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset = ind2)
            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(s, W, trans = 'T')
        helpers.sp_create("unscale_s", 5600)

        blas.copy(lmbda, z, n = mnl + dims['l'] + sum(dims['q']))
        ind = mnl + dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset = ind2)
            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(z, W, inverse = 'I')
        helpers.sp_create("unscale_z", 5700)

        gap = blas.dot(lmbda, lmbda) 



def cp(F, G = None, h = None, dims = None, A = None, b = None,
    kktsolver = None, xnewcopy = None, xdot = None, xaxpy = None,
    xscal = None, ynewcopy = None, ydot = None, yaxpy = None, 
    yscal = None):

    """
    Solves a convex optimization problem
    
        minimize    f0(x)
        subject to  fk(x) <= 0, k = 1, ..., mnl
                    G*x   <= h
                    A*x   =  b.                      

    f = (f0, f1, ..., fmnl) is convex and twice differentiable.  The linear
    inequalities are with respect to a cone C defined as the Cartesian 
    product of N + M + 1 cones:
    
        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.  The 
    next N cones are second order cones of dimension mq[0], ..., mq[N-1].
    The second order cone of dimension m is defined as
    
        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.  


    """

    import math 
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix 

    mnl, x0 = F()

    # Argument error checking depends on level of customization.
    customkkt = type(kktsolver) is not str
    operatorG = G is not None and type(G) not in (matrix, spmatrix)
    operatorA = A is not None and type(A) not in (matrix, spmatrix)
    if (operatorG or operatorA) and not customkkt:
        raise ValueError("use of function valued G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or
        xscal != None)
    if customx and (not operatorG or not operatorA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or 
        yscal != None)
    if customy and (not operatorA or not customkkt):
        raise ValueError("use of non vector type for y requires "\
            "function valued A and user-provided kktsolver")

    if not customx:  
        if type(x0) is not matrix or x0.typecode != 'd' or x0.size[1] != 1:
            raise TypeError("'x0' must be a 'd' matrix with one column")
        
    if h is None: h = matrix(0.0, (0,1))
    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with one column")
    if not dims: dims = {'l': h.size[0], 'q': [], 's': []}

    # Dimension of the product cone of the linear inequalities. with 's' 
    # components unpacked.
    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    if G is None:
        if customx:
            def G(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            G = spmatrix([], [], [], (0, x0.size[0]))
    if type(G) is matrix or type(G) is spmatrix:
        if G.typecode != 'd' or G.size != (cdim, x0.size[0]):
            raise TypeError("'G' must be a 'd' matrix with size (%d, %d)"\
                %(cdim, x0.size[0]))
        def fG(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha, 
                beta = beta)
    else:
        fG = G

    if A is None:
        if customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, x0.size[0]))
    if type(A) is matrix or type(A) is spmatrix:
        if A.typecode != 'd' or A.size[1] != x0.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns" \
                %x0.size[0])
        def fA(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        fA = A
    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if not operatorA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    if b is None and customy:  
        raise ValueEror("use of non vector type for y requires b")


    if xnewcopy is None: xnewcopy = matrix 
    if xdot is None: xdot = blas.dot
    if xaxpy is None: xaxpy = blas.axpy 
    if xscal is None: xscal = blas.scal 
    def xcopy(x, y): 
        xscal(0.0, y) 
        xaxpy(x, y)
    if ynewcopy is None: ynewcopy = matrix 
    if ydot is None: ydot = blas.dot 
    if yaxpy is None: yaxpy = blas.axpy 
    if yscal is None: yscal = blas.scal
    def ycopy(x, y): 
        yscal(0.0, y) 
        yaxpy(x, y)
             

    # The problem is solved by applying cpl() to the epigraph form 
    #
    #     minimize   t 
    #     subject to f0(x) - t <= 0
    #                f1(x) <= 0
    #                ...
    #                fmnl(x) <= 0
    #                G*x <= h
    #                A*x = b.
    #
    # The epigraph form variable is stored as a list [x, t].

    # Epigraph form objective c = (0, 1).
    c = [ xnewcopy(x0), 1 ] 
    xscal(0.0, c[0])

    # Nonlinear inequalities for the epigraph problem
    #
    #     f_e(x,t) = (f0(x) - t, f1(x), ..., fmnl(x)).
    #     

    def F_e(x = None, z = None):

        if x is None: 
            return mnl+1, [ x0, 0.0 ]

        else:
            if z is None:
                v = F(x[0])
                if v is None or v[0] is None: return None, None
                val = matrix(v[0], tc = 'd')
                val[0] -= x[1]
                Df = v[1]
            else:
                val, Df, H = F(x[0], z)
                val = matrix(val, tc = 'd')
                val[0] -= x[1]

            if type(Df) in (matrix, spmatrix):
                def Df_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):  
                    if trans == 'N':
                        #print "Df_e N: df=\n", helpers.str2(Df, "%.7f")
                        #print "Df_e N: u=\n", helpers.str2(u[0], "%.7f")
                        #print "Df_e N: v=\n", helpers.str2(v, "%.7f")
                        base.gemv(Df, u[0], v, alpha = alpha, beta = beta,
                            trans = 'N')
                        v[0] -= alpha * u[1]
                        #print "Df_e N: v 1 =\n", helpers.str2(v, "%.7f")
                    else:
                        #print "Df_e T: df=\n", helpers.str2(Df, "%.7f")
                        #print "Df_e T: u=\n", helpers.str2(u, "%.7f")
                        #print "Df_e T: v=\n", helpers.str2(v[0], "%.7f")
                        base.gemv(Df, u, v[0], alpha = alpha, beta = beta,
                            trans = 'T')
                        #print "Df_e T: v 1 =\n", helpers.str2(v[0], "%.7f")
                        v[1] = -alpha * u[0] + beta * v[1]
            else:
                def Df_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):  
                    if trans == 'N':
                        Df(u[0], v, alpha = alpha, beta = beta, 
                            trans = 'N')
                        v[0] -= alpha * u[1]
                    else:
                        Df(u, v[0], alpha = alpha, beta = beta, 
                            trans = 'T')
                        v[1] = -alpha * u[0] + beta * v[1]

            if z is None:
                return val, Df_e
            else:
                if type(H) in (matrix, spmatrix):
                    def H_e(u, v, alpha = 1.0, beta = 1.0):
                        #print "H_e:\n", helpers.str2(H, "%.3f")
                        base.symv(H, u[0], v[0], alpha = alpha, 
                            beta = beta) 
                        v[1] += beta*v[1]
                else:
                    def H_e(u, v, alpha = 1.0, beta = 1.0):
                        H(u[0], v[0], alpha = alpha, beta = beta)
                        v[1] += beta*v[1]
                return val, Df_e, H_e


    # Linear inequality constraints.
    #
    #     G_e  = [ G, 0 ]
    #

    if type(G) in (matrix, spmatrix):
        def G_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                misc.sgemv(G, u[0], v, dims, alpha = alpha, beta = beta) 
            else:
                misc.sgemv(G, u, v[0], dims, alpha = alpha, beta = beta, 
                    trans = 'T') 
                v[1] *= beta
    else:
        def G_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                G(u[0], v, alpha = alpha, beta = beta) 
            else:
                G(u, v[0], alpha = alpha, beta = beta, trans = 'T') 
                v[1] *= beta


    # Linear equality constraints.
    #
    #     A_e = [ A, 0 ]
    #

    if type(A) in (matrix, spmatrix):
        def A_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                base.gemv(A, u[0], v, alpha = alpha, beta = beta) 
            else:
                base.gemv(A, u, v[0], alpha = alpha, beta = beta, 
                    trans = 'T') 
                v[1] *= beta
    else:
        def A_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                A(u[0], v, alpha = alpha, beta = beta) 
            else:
                A(u, v[0], alpha = alpha, beta = beta, trans = 'T') 
                v[1] *= beta
 

    # kktsolver(x, z, W) returns a routine for solving equations with 
    # coefficient matrix
    #
    #         [ H             A'   [Df[1:]; G]' ]
    #     K = [ A             0    0            ]. 
    #         [ [Df[1:]; G]   0    -W'*W        ]
 
    if kktsolver is None: 
        if dims and (dims['q'] or dims['s']):  
            kktsolver = 'chol'            
        else:
            kktsolver = 'chol2'            
    if kktsolver in ('ldl', 'chol', 'chol2', 'qr'):
        if kktsolver == 'ldl':
            factor = localmisc.kkt_ldl(G, dims, A, mnl)
        elif kktsolver == 'qr':
            factor = localmisc.kkt_qr(G, dims, A, mnl)
        elif kktsolver == 'chol':
            factor = localmisc.kkt_chol(G, dims, A, mnl)
        else: 
            factor = localmisc.kkt_chol2(G, dims, A, mnl)
        def kktsolver(x, z, W):
            f, Df, H = F(x, z)
            return factor(W, H, Df[1:,:])             

    ux, uz = xnewcopy(x0), matrix(0.0, (mnl + cdim, 1))
    def kktsolver_e(x, znl, W):

        We = W.copy()
        We['dnl'] = W['dnl'][1:]
        We['dnli'] = W['dnli'][1:]
        #helpers.printW(We)
        g = kktsolver(x[0], znl, We)

        f, Df = F(x[0])
        if type(Df) is matrix or type(Df) is spmatrix:
            gradf0 = Df[0,:].T
        else:
            gradf0 = xnewcopy(x[0])
            e0 = matrix(0.0, (mnl + 1, 1))
            e0[0] = 1.0
            Df(e0, gradf0, trans = 'T')
        #print "kktsolve_e gradf0=\n", helpers.str2(gradf0, "%.17f")

        def solve(x, y, z):

            # Solves 
            #
            #    [ [ H   0  ]   [ A' ]  [ Df'  G'] ] [ ux ]    [ bx ]
            #    [ [ 0   0  ]   [ 0  ]  [ -e0' 0 ] ] [    ]    [    ]
            #    [                                 ] [    ]    [    ]
            #    [ [ A   0  ]   0       0          ] [ uy ] =  [ by ].
            #    [                                 ] [    ]    [    ]
            #    [ [ Df -e0 ]   0       -W'*W      ] [ uz ]    [ bz ]
            #    [ [ G   0  ]                      ] [    ]    [    ]
            # 
            # The solution is:
            #
            #     uz[0] = -bx[1] 
            #
            #     [ ux[0]  ]          [ bx[0] + bx[1] * gradf0 ]
            #     [ uy     ] = K^-1 * [ by                     ].
            #     [ uz[1:] ]          [ bz[1:]                 ]
            #
            #     ux[1] = gradf0' * ux[0] - W['dnl'][0]**2 * uz[0] - bz[0]
            #           = gradf0' * ux[0] + W['dnl'][0]**2 * bx[1] - bz[0].
            #
            # Instead of uz we return the scaled solution W*uz.

            a = z[0]
            #print "solve_e 0 a=%.7f x=\n" % a, helpers.str2(x[0], "%.17f")
            #print "solve_e 0 z=\n", helpers.str2(z, "%.17f")
            xcopy(x[0], ux)
            xaxpy(gradf0, ux, alpha = x[1])
            blas.copy(z, uz, offsetx = 1)
            #print "solve pre-g: uz=\n", helpers.str2(uz, "%.7f")
            g(ux, y, uz)
            #print "solve post-g: ux=\n", helpers.str2(ux, "%.7f")
            #print "solve post-g: uz=\n", helpers.str2(uz, "%.7f")
            z[0] = -x[1] * W['dnl'][0]
            blas.copy(uz, z, offsety = 1)
            xcopy(ux, x[0])
            x[1] = xdot(gradf0, x[0]) + W['dnl'][0]**2 * x[1] - a
            #print "solve_e 1 x=\n", helpers.str2(x, "%.17f")
            #print "solve_e 1 z=\n", helpers.str2(z, "%.17f")
            
        return solve

    def xnewcopy_e(x):
        return [ xnewcopy(x[0]), x[1] ]

    def xdot_e(x, y):
        return xdot(x[0], y[0]) + x[1]*y[1]

    def xaxpy_e(x, y, alpha = 1.0):
        xaxpy(x[0], y[0], alpha = alpha)
        y[1] += alpha*x[1]

    def xscal_e(alpha, x):
        xscal(alpha, x[0])
        x[1] *= alpha

    sol = cpl(c, F_e, G_e, h, dims, A_e, b, kktsolver_e, xnewcopy_e, 
         xdot_e, xaxpy_e, xscal_e)

    sol['x'] = sol['x'][0]
    sol['znl'], sol['snl'] = sol['znl'][1:], sol['snl'][1:]
    return sol


def gp(K, F, g, G=None, h=None, A=None, b=None, kktsolver=None):

    """
    Solves a geometric program

        minimize    log sum exp (F0*x+g0)
        subject to  log sum exp (Fi*x+gi) <= 0,  i=1,...,m
                    G*x <= h      
                    A*x = b
    """

    import math 
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix 

    if type(K) is not list or [ k for k in K if type(k) is not int 
        or k <= 0 ]:
        raise TypeError("'K' must be a list of positive integers")
    mnl = len(K)-1
    l = sum(K)

    if type(F) not in (matrix, spmatrix) or F.typecode != 'd' or \
        F.size[0] != l:
        raise TypeError("'F' must be a dense or sparse 'd' matrix "\
            "with %d rows" %l)
    if type(g) is not matrix or g.typecode != 'd' or g.size != (l,1): 
        raise TypeError("'g' must be a dene 'd' matrix of "\
            "size (%d,1)" %l)
    n = F.size[1]

    if G is None: G = spmatrix([], [], [], (0,n))
    if h is None: h = matrix(0.0, (0,1))
    if type(G) not in (matrix, spmatrix) or G.typecode != 'd' or \
        G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = G.size[0]
    if type(h) is not matrix or h.typecode != 'd' or h.size != (ml,1):
        raise TypeError("'h' must be a dense 'd' matrix of "\
            "size (%d,1)" %ml)
    dims = {'l': ml, 's': [], 'q': []}

    if A is None: A = spmatrix([], [], [], (0,n))
    if b is None: b = matrix(0.0, (0,1))
    if type(A) not in (matrix, spmatrix) or A.typecode != 'd' or \
        A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense 'd' matrix of "\
            "size (%d,1)" %p)

    y = matrix(0.0, (l,1))
    u = matrix(0.0, (max(K),1))
    Fsc = matrix(0.0, (max(K),n))

    cs1 = [ sum(K[:i]) for i in range(mnl+1) ] 
    cs2 = [ cs1[i] + K[i] for i in range(mnl+1) ]
    ind = list(zip(range(mnl+1), cs1, cs2))

    def Fgp(x = None, z = None):

        if x is None: return mnl, matrix(0.0, (n,1))
	
        f = matrix(0.0, (mnl+1,1))
        Df = matrix(0.0, (mnl+1,n))

        # y = F*x+g
        blas.copy(g, y)
        base.gemv(F, x, y, beta=1.0)
        #print "y=\n", helpers.str2(y, "%.3f")

        if z is not None: H = matrix(0.0, (n,n))

        for i, start, stop in ind:

            #print "start, stop = %d, %d" %(start, stop)

            # yi := exp(yi) = exp(Fi*x+gi) 
            ymax = max(y[start:stop])
            y[start:stop] = base.exp(y[start:stop] - ymax)

            # fi = log sum yi = log sum exp(Fi*x+gi)
            ysum = blas.asum(y, n=stop-start, offset=start)
            f[i] = ymax + math.log(ysum)
            #print "ymax, ysum = %.3f, %.3f" %(ymax, ysum)

            # yi := yi / sum(yi) = exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            blas.scal(1.0/ysum, y, n=stop-start, offset=start)

            # gradfi := Fi' * yi 
            #        = Fi' * exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            base.gemv(F, y, Df, trans='T', m=stop-start, incy=mnl+1,
                offsetA=start, offsetx=start, offsety=i)

            if z is not None:

                # Hi = Fi' * (diag(yi) - yi*yi') * Fi 
                #    = Fisc' * Fisc
                # where 
                # Fisc = diag(yi)^1/2 * (I - 1*yi') * Fi
                #      = diag(yi)^1/2 * (Fi - 1*gradfi')

                Fsc[:K[i], :] = F[start:stop, :] 
                #print "Fsc [%d rows] =\n" % Fsc.size[0], helpers.str2(Fsc, "%.3f")
                for k in range(start,stop):
                   blas.axpy(Df, Fsc, n=n, alpha=-1.0, incx=mnl+1,
                       incy=Fsc.size[0], offsetx=i, offsety=k-start)
                   blas.scal(math.sqrt(y[k]), Fsc, inc=Fsc.size[0],
                       offset=k-start)

                #print "Fsc =\n", helpers.str2(Fsc, "%.3f")
                # H += z[i]*Hi = z[i] * Fisc' * Fisc
                blas.syrk(Fsc, H, trans='T', k=stop-start, alpha=z[i],
                    beta=1.0)

        if z is None:
            #print "Df=\n", helpers.str2(Df, "%.3f")
            return f, Df
        else:
            #print "Df=\n", helpers.str2(Df, "%.3f")
            #print "H=\n", helpers.str2(H, "%.3f")
            return f, Df, H

    return cp(Fgp, G, h, dims, A, b, kktsolver=kktsolver)
