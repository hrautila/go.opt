// This file is part of go.opt package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package main

import (
    "flag"
    "fmt"
    "github.com/hrautila/cvx"
    "github.com/hrautila/cvx/checkpnt"
    "github.com/hrautila/linalg/blas"
    "github.com/hrautila/matrix"
)

var xVal, sVal, zVal string
var spPath string
var maxIter int
var spVerbose bool
var solver string

func init() {
    flag.BoolVar(&spVerbose, "V", false, "Savepoint verbose reporting.")
    flag.IntVar(&maxIter, "N", -1, "Max number of iterations.")
    flag.StringVar(&spPath, "sp", "", "savepoint directory")
    flag.StringVar(&solver, "solver", "", "Solver name")
    flag.StringVar(&xVal, "x", "", "Reference value for X")
    flag.StringVar(&sVal, "s", "", "Reference value for S")
    flag.StringVar(&zVal, "z", "", "Reference value for Z")
}

func error(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
    diff = ref.Minus(val)
    nrm = blas.Nrm2(diff).Float()
    return
}

func check(x, s, z *matrix.FloatMatrix) {
    if len(xVal) > 0 {
        ref, _ := matrix.FloatParse(xVal)
        nrm, diff := error(ref, x)
        fmt.Printf("x: nrm=%.9f\n", nrm)
        if nrm > 10e-7 {
            fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
        }
    }
    if len(sVal) > 0 {
        ref, _ := matrix.FloatParse(sVal)
        nrm, diff := error(ref, s)
        fmt.Printf("s: nrm=%.9f\n", nrm)
        if nrm > 10e-7 {
            fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
        }
    }
    if len(zVal) > 0 {
        ref, _ := matrix.FloatParse(zVal)
        nrm, diff := error(ref, z)
        fmt.Printf("z: nrm=%.9f\n", nrm)
        if nrm > 10e-7 {
            fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
        }
    }
}

func main() {
    flag.Parse()
    if len(spPath) > 0 {
        checkpnt.Reset(spPath)
        checkpnt.Activate()
        checkpnt.Verbose(spVerbose)
        checkpnt.Format("%.17f")
    }

    A := matrix.FloatNew(2, 3, []float64{1.0, -1.0, 0.0, 1.0, 0.0, 1.0})
    b := matrix.FloatNew(2, 1, []float64{1.0, 0.0})
    c := matrix.FloatNew(3, 1, []float64{0.0, 1.0, 0.0})
    G := matrix.FloatNew(1, 3, []float64{0.0, -1.0, 1.0})
    h := matrix.FloatNew(1, 1, []float64{0.0})
    //dims := sets.NewDimensionSet("l", "q", "s")
    //dims.Set("l", []int{1})

    fmt.Printf("A=\n%v\n", A)
    fmt.Printf("b=\n%v\n", b)
    fmt.Printf("G=\n%v\n", G)
    fmt.Printf("h=\n%v\n", h)
    fmt.Printf("c=\n%v\n", c)

    var solopts cvx.SolverOptions
    solopts.MaxIter = 30
    solopts.ShowProgress = true
    if maxIter > -1 {
        solopts.MaxIter = maxIter
    }
    if len(solver) > 0 {
        solopts.KKTSolverName = solver
    }
    sol, err := cvx.Lp(c, G, h, A, b, &solopts, nil, nil)
    if sol != nil && sol.Status == cvx.Optimal {
        x := sol.Result.At("x")[0]
        s := sol.Result.At("s")[0]
        z := sol.Result.At("z")[0]
        fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
        fmt.Printf("s=\n%v\n", s.ToString("%.9f"))
        fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
        check(x, s, z)
    } else {
        fmt.Printf("status: %v\n", err)
    }
}

// Local Variables:
// tab-width: 4
// End:
