// This file is part of go.opt package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package main

import (
    "flag"
    "fmt"
    "github.com/hrautila/cvx"
    "github.com/hrautila/cvx/checkpnt"
    "github.com/hrautila/cvx/sets"
    "github.com/hrautila/linalg"
    "github.com/hrautila/linalg/blas"
    "github.com/hrautila/linalg/lapack"
    "github.com/hrautila/matrix"
)

var xVal, zVal, AVal, bVal string
var spPath string
var maxIter int
var spVerbose bool
var solver string

// helper variables for checkpointing...
var loopg, loopf int

func init() {
    flag.BoolVar(&spVerbose, "V", false, "Savepoint verbose reporting.")
    flag.IntVar(&maxIter, "N", 30, "Max number of iterations.")
    flag.StringVar(&spPath, "sp", "", "savepoint directory")
    flag.StringVar(&solver, "solver", "", "Solver name")
    flag.StringVar(&xVal, "x", "", "Reference value for X")
    flag.StringVar(&zVal, "z", "", "Reference value for Z")
    flag.StringVar(&AVal, "A", "", "Problem data A")
    flag.StringVar(&bVal, "b", "", "Problem data b")
    loopg = 0
    loopf = 0
}

func errorToRef(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
    diff = matrix.Minus(ref, val)
    nrm = blas.Nrm2(diff).Float()
    return
}

func check(x, z *matrix.FloatMatrix) {
    if len(xVal) > 0 {
        ref, _ := matrix.FloatParse(xVal)
        nrm, diff := errorToRef(ref, x)
        fmt.Printf("x: nrm=%.9f\n", nrm)
        if nrm > 10e-7 {
            fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
        }
    }
    if len(zVal) > 0 {
        ref, _ := matrix.FloatParse(zVal)
        nrm, diff := errorToRef(ref, z)
        fmt.Printf("z: nrm=%.9f\n", nrm)
        if nrm > 10e-7 {
            fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
        }
    }
}

// Internal type for MatrixG interface
type matrixFs struct {
    A *matrix.FloatMatrix
}

func (g *matrixFs) Gf(x, y *matrix.FloatMatrix, alpha, beta float64, trans linalg.Option) error {
    //
    minor := 0
    if !checkpnt.MinorEmpty() {
        minor = checkpnt.MinorTop()
    } else {
        loopg += 1
        minor = loopg
    }
    checkpnt.Check("00-Gfunc", minor)

    m, n := g.A.Size()
    y.Scale(beta)

    // x_n = x[:n]
    //x_n := matrix.FloatVector(x.FloatArray()[:n])
    x_n := x.SubMatrix(0, 0, n, 1).Copy()

    // x_n_2n = x[n:2*n]
    //x_n_2n := matrix.FloatVector(x.FloatArray()[n : 2*n])
    x_n_2n := x.SubMatrix(n, 0, n, 1).Copy()

    if linalg.Equal(trans, linalg.OptNoTrans) {
        // y += alpha * G * x

        // y[:n] += alpha * (x[:n] - x[n:2*n])
        y_n := matrix.Minus(x_n, x_n_2n).Scale(alpha)
        y.SubMatrix(0, 0, n, 1).Plus(y_n)
        //y.AddIndexes(matrix.Indexes(n), y_n.FloatArray())

        // y[n:2*n] += alpha * (-x[:n] - x[n:2*n]) = -alpha * (x[:n]+x[n:2*n])
        y_n = matrix.Plus(x_n, x_n_2n).Scale(-alpha)
        y.SubMatrix(n, 0, n, 1).Plus(y_n)
        //y.AddIndexes(matrix.Indexes(n, 2*n), y_n.FloatArray())

        // y[2*n+1:] += -alpha * A * x[:n]
        y_2n := matrix.Times(g.A, x_n).Scale(-alpha)
        //y.AddIndexes(matrix.Indexes(2*n+1, y.NumElements()), y_2n.FloatArray())
        y.SubMatrix(2*n+1, 0, y_2n.NumElements(), 1).Plus(y_2n)
    } else {
        // x_m = x[-m:]
        //x_m := matrix.FloatVector(x.FloatArray()[x.NumElements()-m:])
        x_m := x.SubMatrix(x.NumElements()-m, 0)

        // x_tmp = (x[:n] - x[n:2*n] - A.T * x[-m:])
        x_tmp := matrix.Minus(x_n, x_n_2n, matrix.Times(g.A.Transpose(), x_m))

        // y[:n] += alpha * (x[:n] - x[n:2*n] - A.T * x[-m:])
        //y.AddIndexes(matrix.Indexes(n), x_tmp.Scale(alpha).FloatArray())
        y.SubMatrix(0, 0, n, 1).Plus(x_tmp.Scale(alpha))

        x_tmp = matrix.Plus(x_n, x_n_2n).Scale(-alpha)
        //y.AddIndexes(matrix.Indexes(n, y.NumElements()), x_tmp.FloatArray())
        y.SubMatrix(n, 0).Plus(x_tmp)
    }
    checkpnt.Check("10-Gfunc", minor)
    return nil
}

func qcl1(A, b *matrix.FloatMatrix) (*cvx.Solution, error) {

    // Returns the solution u, z of
    //
    //   (primal)  minimize    || u ||_1       
    //             subject to  || A * u - b ||_2  <= 1
    //
    //   (dual)    maximize    b^T z - ||z||_2
    //             subject to  || A'*z ||_inf <= 1.
    //
    // Exploits structure, assuming A is m by n with m >= n. 

    m, n := A.Size()
    Fkkt := func(W *sets.FloatMatrixSet) (f cvx.KKTFunc, err error) {

        minor := 0
        if !checkpnt.MinorEmpty() {
            minor = checkpnt.MinorTop()
        }

        err = nil
        f = nil
        beta := W.At("beta")[0].GetIndex(0)
        v := W.At("v")[0]

        // As = 2 * v *(v[1:].T * A)
        //v_1 := matrix.FloatNew(1, v.NumElements()-1, v.FloatArray()[1:])
        v_1 := v.SubMatrix(1, 0).Transpose()

        As := matrix.Times(v, matrix.Times(v_1, A)).Scale(2.0)

        //As_1 := As.GetSubMatrix(1, 0, m, n)
        //As_1.Scale(-1.0)
        //As.SetSubMatrix(1, 0, matrix.Minus(As_1, A))
        As_1 := As.SubMatrix(1, 0, m, n)
        As_1.Scale(-1.0)
        As_1.Minus(A)
        As.Scale(1.0 / beta)

        S := matrix.Times(As.Transpose(), As)
        checkpnt.AddMatrixVar("S", S)

        d1 := W.At("d")[0].SubMatrix(0, 0, n, 1).Copy()
        d2 := W.At("d")[0].SubMatrix(n, 0).Copy()

        // D = 4.0 * (d1**2 + d2**2)**-1
        d := matrix.Plus(matrix.Mul(d1, d1), matrix.Mul(d2, d2)).Inv().Scale(4.0)
        // S[::n+1] += d
        S.Diag().Plus(d.Transpose())

        err = lapack.Potrf(S)
        checkpnt.Check("00-Fkkt", minor)
        if err != nil {
            return
        }

        f = func(x, y, z *matrix.FloatMatrix) (err error) {

            minor := 0
            if !checkpnt.MinorEmpty() {
                minor = checkpnt.MinorTop()
            } else {
                loopf += 1
                minor = loopf
            }
            checkpnt.Check("00-f", minor)

            // -- z := - W**-T * z 
            // z[:n] = -div( z[:n], d1 )
            z_val := z.SubMatrix(0, 0, n, 1)
            z_res := matrix.Div(z_val, d1).Scale(-1.0)
            z.SubMatrix(0, 0, n, 1).Set(z_res)

            // z[n:2*n] = -div( z[n:2*n], d2 )
            z_val = z.SubMatrix(n, 0, n, 1)
            z_res = matrix.Div(z_val, d2).Scale(-1.0)
            z.SubMatrix(n, 0, n, 1).Set(z_res)

            // z[2*n:] -= 2.0*v*( v[0]*z[2*n] - blas.dot(v[1:], z[2*n+1:]) ) 
            v0_z2n := v.GetIndex(0) * z.GetIndex(2*n)
            v1_z2n := blas.DotFloat(v, z, &linalg.IOpt{"offsetx", 1}, &linalg.IOpt{"offsety", 2*n + 1})
            z_res = matrix.Scale(v, -2.0*(v0_z2n-v1_z2n))
            z.SubMatrix(2*n, 0, z_res.NumElements(), 1).Plus(z_res)

            // z[2*n+1:] *= -1.0
            z.SubMatrix(2*n+1, 0).Scale(-1.0)

            // z[2*n:] /= beta
            z.SubMatrix(2*n, 0).Scale(1.0 / beta)

            // -- x := x - G' * W**-1 * z

            // z_n = z[:n], z_2n = z[n:2*n], z_m = z[-(m+1):], 
            z_n := z.SubMatrix(0, 0, n, 1)
            z_2n := z.SubMatrix(n, 0, n, 1)
            z_m := z.SubMatrix(z.NumElements()-(m+1), 0)

            // x[:n] -= div(z[:n], d1) - div(z[n:2*n], d2) + As.T * z[-(m+1):]
            z_res = matrix.Minus(matrix.Div(z_n, d1), matrix.Div(z_2n, d2))
            a_res := matrix.Times(As.Transpose(), z_m)
            z_res.Plus(a_res).Scale(-1.0)
            x.SubMatrix(0, 0, n, 1).Plus(z_res)

            // x[n:] += div(z[:n], d1) + div(z[n:2*n], d2) 
            z_res = matrix.Plus(matrix.Div(z_n, d1), matrix.Div(z_2n, d2))
            x.SubMatrix(n, 0, z_res.NumElements(), 1).Plus(z_res)
            checkpnt.Check("15-f", minor)

            // Solve for x[:n]:
            //
            //    S*x[:n] = x[:n] - (W1**2 - W2**2)(W1**2 + W2**2)^-1 * x[n:]

            // w1 = (d1**2 - d2**2), w2 = (d1**2 + d2**2)
            w1 := matrix.Minus(matrix.Mul(d1, d1), matrix.Mul(d2, d2))
            w2 := matrix.Plus(matrix.Mul(d1, d1), matrix.Mul(d2, d2))

            // x[:n] += -mul( div(w1, w2), x[n:])
            x_n := x.SubMatrix(n, 0)
            x_val := matrix.Mul(matrix.Div(w1, w2), x_n).Scale(-1.0)
            x.SubMatrix(0, 0, n, 1).Plus(x_val)
            checkpnt.Check("25-f", minor)

            // Solve for x[n:]:
            //
            //    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]

            err = lapack.Potrs(S, x)
            if err != nil {
                fmt.Printf("Potrs error: %s\n", err)
            }
            checkpnt.Check("30-f", minor)

            // Solve for x[n:]:
            //
            //    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]

            // w1 = (d1**-2 - d2**-2), w2 = (d1**-2 + d2**-2)
            w1 = matrix.Minus(matrix.Mul(d1, d1).Inv(), matrix.Mul(d2, d2).Inv())
            w2 = matrix.Plus(matrix.Mul(d1, d1).Inv(), matrix.Mul(d2, d2).Inv())
            x_n = x.SubMatrix(0, 0, n, 1)

            // x[n:] += mul( d1**-2 - d2**-2, x[:n])
            x_val = matrix.Mul(w1, x_n)
            x.SubMatrix(n, 0, x_val.NumElements(), 1).Plus(x_val)
            checkpnt.Check("35-f", minor)

            // x[n:] = div( x[n:], d1**-2 + d2**-2)
            x_n = x.SubMatrix(n, 0)
            x_val = matrix.Div(x_n, w2)
            x.SubMatrix(n, 0, x_val.NumElements(), 1).Set(x_val)
            checkpnt.Check("40-f", minor)

            // x_n = x[:n], x-2n = x[n:2*n]
            x_n = x.SubMatrix(0, 0, n, 1)
            x_2n := x.SubMatrix(n, 0, n, 1)

            // z := z + W^-T * G*x 
            // z[:n] += div( x[:n] - x[n:2*n], d1) 
            x_val = matrix.Div(matrix.Minus(x_n, x_2n), d1)
            z.SubMatrix(0, 0, n, 1).Plus(x_val)
            checkpnt.Check("44-f", minor)

            // z[n:2*n] += div( -x[:n] - x[n:2*n], d2) 
            x_val = matrix.Div(matrix.Plus(x_n, x_2n).Scale(-1.0), d2)
            z.SubMatrix(n, 0, n, 1).Plus(x_val)
            checkpnt.Check("48-f", minor)

            // z[2*n:] += As*x[:n]
            x_val = matrix.Times(As, x_n)
            z.SubMatrix(2*n, 0, x_val.NumElements(), 1).Plus(x_val)

            checkpnt.Check("50-f", minor)

            return nil
        }
        return
    }

    // matrix(n*[0.0] + n*[1.0])
    c := matrix.FloatZeros(2*n, 1)
    c.SubMatrix(n, 0).SetIndexes(1.0)

    h := matrix.FloatZeros(2*n+m+1, 1)
    h.SetIndexes(1.0, 2*n)
    // h[2*n+1:] = -b
    h.SubMatrix(2*n+1, 0).Plus(b).Scale(-1.0)
    G := &matrixFs{A}

    dims := sets.DSetNew("l", "q", "s")
    dims.Set("l", []int{2 * n})
    dims.Set("q", []int{m + 1})

    var solopts cvx.SolverOptions
    solopts.ShowProgress = true
    if maxIter > 0 {
        solopts.MaxIter = maxIter
    }
    if len(solver) > 0 {
        solopts.KKTSolverName = solver
    }
    return cvx.ConeLpCustomMatrix(c, G, h, nil, nil, dims, Fkkt, &solopts, nil, nil)
}

func main() {

    var A, b *matrix.FloatMatrix = nil, nil
    m, n := 20, 20

    blas.PanicOnError(true)
    matrix.PanicOnError(true)

    flag.Parse()
    if len(spPath) > 0 {
        checkpnt.Reset(spPath)
        checkpnt.Activate()
        checkpnt.Verbose(spVerbose)
        checkpnt.Format("%.17f")
    }
    if len(AVal) > 0 {
        A, _ = matrix.FloatParse(AVal)
        if A == nil {
            fmt.Printf("could not parse:\n%s\n", AVal)
            return
        }
    } else {
        A = matrix.FloatNormal(m, n)
    }
    if len(bVal) > 0 {
        b, _ = matrix.FloatParse(bVal)
        if b == nil {
            fmt.Printf("could not parse:\n%s\n", bVal)
            return
        }
    } else {
        b = matrix.FloatNormal(m, 1)
    }

    sol, err := qcl1(A, b)
    if sol != nil {
        r := sol.Result.At("x")[0]
        x := matrix.FloatVector(r.FloatArray()[:A.Cols()])
        r = sol.Result.At("z")[0]
        z := matrix.FloatVector(r.FloatArray()[r.NumElements()-A.Rows():])
        fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
        fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
        check(x, z)
    } else {
        fmt.Printf("status: %v\n", err)
    }
}

// Local Variables:
// tab-width: 4
// End:
