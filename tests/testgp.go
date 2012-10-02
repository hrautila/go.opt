
// This file is part of go.opt package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package main

import (
	"github.com/hrautila/matrix"
	"github.com/hrautila/linalg/blas"
	"github.com/hrautila/cvx"
	"github.com/hrautila/cvx/checkpnt"
	"fmt"
	"flag"
)

var xVal string
var spPath string
var spVerbose bool
var maxIter int
var solver string

func init() {
	flag.BoolVar(&spVerbose, "V", false, "Savepoint verbose reporting.")
	flag.IntVar(&maxIter, "N", -1, "Max number of iterations.")
	flag.StringVar(&spPath, "sp", "", "savepoint directory")
	flag.StringVar(&solver, "solver", "", "Solver name")
	flag.StringVar(&xVal, "x", "", "Reference value for X")
}
	
func error(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = ref.Minus(val)
	nrm = blas.Nrm2(diff).Float()
	return
}

func check(x *matrix.FloatMatrix) {
	if len(xVal) > 0 {
		ref, _ := matrix.FloatParse(xVal)
		nrm, diff := error(ref, x)
		fmt.Printf("x: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
}

func main() {
	flag.Parse()

	aflr := 1000.0
	awall := 100.0
	alpha := 0.5
	beta := 2.0
	gamma := 0.5
	delta := 2.0

	fdata := [][]float64{
		[]float64{-1.0, 1.0,  1.0,  0.0, -1.0,  1.0,  0.0,  0.0},
		[]float64{-1.0, 1.0,  0.0,  1.0,  1.0, -1.0,  1.0, -1.0},
		[]float64{-1.0, 0.0,  1.0,  1.0,  0.0,  0.0, -1.0,  1.0}}

	gdata := []float64{1.0, 2.0/awall, 2.0/awall, 1.0/aflr, alpha, 1.0/beta, gamma, 1.0/delta}
	
	g := matrix.FloatNew(8, 1, gdata).Log()
	F := matrix.FloatMatrixFromTable(fdata)
	K := []int{1, 2, 1, 1, 1, 1, 1}

	var solopts cvx.SolverOptions
	solopts.MaxIter = 40
	if maxIter > 0 {
		solopts.MaxIter = maxIter
	}
	if len(spPath) > 0 {
		checkpnt.Reset(spPath)
		checkpnt.Activate()
		checkpnt.Verbose(spVerbose)
		checkpnt.Format("%.7f")
	}
	solopts.ShowProgress = true
	if maxIter > 0 {
		solopts.MaxIter = maxIter
	}
	if len(solver) > 0 {
		solopts.KKTSolverName = solver
	}
	sol, err := cvx.Gp(K, F, g, nil, nil, nil, nil, &solopts)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		r := matrix.Exp(x)
		h := r.GetIndex(0)
		w := r.GetIndex(1)
		d := r.GetIndex(2)
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
        fmt.Printf("\n h = %f,  w = %f, d = %f.\n", h, w, d)   
		check(x)
	} else {
		fmt.Printf("status: %v\n", err)
	}
}

// Local Variables:
// tab-width: 4
// End:
