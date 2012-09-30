//
// cvxopt/examples/book/chap6/cvxfit.py
//
package main

import (
	"github.com/hrautila/matrix"
	"github.com/hrautila/cvx"
	"code.google.com/p/plotinum/vg"
	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"image/color"
	"fmt"
)


var udata []float64 = []float64{
	0.00000000000000000,
	0.04000000000000000,
	0.08000000000000000,
	0.12000000000000000,
	0.16000000000000000,
	0.20000000000000001,
	0.23999999999999999,
	0.28000000000000003,
	0.32000000000000001,
	0.35999999999999999,
	0.40000000000000002,
	0.44000000000000000,
	0.47999999999999998,
	0.52000000000000002,
	0.56000000000000005,
	0.59999999999999998,
	0.64000000000000001,
	0.68000000000000005,
	0.71999999999999997,
	0.76000000000000001,
	0.80000000000000004,
	0.83999999999999997,
	0.88000000000000000,
	0.92000000000000004,
	0.95999999999999996,
	1.00000000000000000,
	1.04000000000000004,
	1.08000000000000007,
	1.12000000000000011,
	1.16000000000000014,
	1.19999999999999996,
	1.23999999999999999,
	1.28000000000000003,
	1.31999999999999984,
	1.35999999999999988,
	1.39999999999999991,
	1.43999999999999995,
	1.47999999999999998,
	1.52000000000000002,
	1.56000000000000005,
	1.60000000000000009,
	1.64000000000000012,
	1.67999999999999994,
	1.71999999999999997,
	1.76000000000000001,
	1.80000000000000004,
	1.84000000000000008,
	1.87999999999999989,
	1.91999999999999993,
	1.95999999999999996,
	2.00000000000000000}

var ydata []float64 = []float64{
	5.20573540028328186,
	5.16852954030387401,
	4.46931747072572261,
	3.16764149229963943,
	3.21867268186792987,
	2.75587741926586371,
	1.63606279181933267,
	0.72527756034765778,
	0.24583926693470126,
	-0.58044829477967541,
	-0.87676552269878805,
	-0.82548372091436673,
	-0.79731422846855127,
	-0.05948396007987816,
	-0.04975524893700162,
	0.70500263843069000,
	0.82600210585526379,
	0.14030590460768733,
	0.51054544095596399,
	0.38582234301290158,
	0.83860085513607274,
	0.41632982151136116,
	0.81154681718230726,
	0.23060126778692916,
	0.84177419098779471,
	0.34454158681673575,
	0.37408513903614865,
	0.86597228912388624,
	0.21207009757657225,
	0.71788999284982635,
	0.80995602827825564,
	1.06910933876018710,
	0.64850168698870958,
	1.09248438768100131,
	0.76143044835863438,
	1.21228570085560228,
	1.17728916334374500,
	0.84659501315421903,
	0.95866894737433894,
	1.82113177659879000,
	1.80159357578572199,
	1.63543886655115456,
	1.77429525508189379,
	2.52647668091466349,
	2.65227763871299915,
	3.75011514867520201,
	4.05642221661261893,
	4.62476810959945528,
	4.91230272885681618,
	5.80689459241500483,
	7.02609346313174044}



func dataset(xs, ys []float64) plotter.XYs {
	pts := make(plotter.XYs, len(xs))
	for i := range xs {
		pts[i].X = xs[i]
		pts[i].Y = ys[i]
	}
	return pts
}

func plotData(name string, us, ys, ts, fs []float64) {
	p, err := plot.New()
	if err != nil {
		fmt.Printf("Cannot create new plot: %s\n", err)
		return
	}
	p.Title.Text = "Least-square fit of convex function"
	p.X.Min = -0.1
	p.X.Max = 2.3
	p.Y.Min = -1.1
	p.Y.Max = 7.2
	p.Add(plotter.NewGrid())

	pts := plotter.NewScatter(dataset(us, ys))
	pts.GlyphStyle.Color = color.RGBA{R:255, A:255}

	fit := plotter.NewLine(dataset(ts, fs))
	fit.LineStyle.Width = vg.Points(1)
	fit.LineStyle.Color = color.RGBA{B:255, A:255}

	p.Add(pts)
	p.Add(fit)
	if err := p.Save(4, 4, name); err != nil {
		fmt.Printf("Save to '%s' failed: %s\n", name, err)
	}
}

func main() {

	m := len(udata)
	nvars := 2*m
	u := matrix.FloatVector(udata[:m])
	y := matrix.FloatVector(ydata[:m])


	// minimize    (1/2) * || yhat - y ||_2^2
	// subject to  yhat[j] >= yhat[i] + g[i]' * (u[j] - u[i]), j, i = 0,...,m-1
	//
	// Variables  yhat (m), g (m).

	P := matrix.FloatZeros(nvars, nvars)
	// set m first diagonal indexes to 1.0
	P.Set(1.0, matrix.DiagonalIndexes(P)[:m]...)
	q := matrix.FloatZeros(nvars, 1)
	q.SetSubMatrix(0, 0, matrix.Scale(y, -1.0))

	// m blocks (i = 0,...,m-1) of linear inequalities 
	//
	//     yhat[i] + g[i]' * (u[j] - u[i]) <= yhat[j], j = 0,...,m-1. 

	G := matrix.FloatZeros(m*m, nvars)
	I := matrix.FloatDiagonal(m, 1.0)

	for i := 0; i < m; i++ {
		// coefficients of yhat[i] (column i)
		G.Set(1.0, matrix.ColumnIndexes(G, i)[i*m:(i+1)*m]...)

		// coefficients of gi[i] (column i, rows i*m ... (i+1)*m)
		rows := matrix.Indexes(i*m, (i+1)*m)
		G.SetAtColumnArray(m+i, rows, matrix.Add(u, -u.GetIndex(i)).FloatArray())

		// coeffients of yhat[i]) from rows i*m ... (i+1)*m, cols 0 ... m
		G.SetSubMatrix(i*m, 0, matrix.Minus(G.GetSubMatrix(i*m, 0, m, m), I))
	}

	h := matrix.FloatZeros(m*m, 1)
	var A, b *matrix.FloatMatrix = nil, nil
	var solopts cvx.SolverOptions
	solopts.ShowProgress = true

	sol, err := cvx.Qp(P, q, G, h, A, b, &solopts, nil)
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	if sol != nil && sol.Status != cvx.Optimal {
		fmt.Printf("status not optimal\n")
		return
	}
	x := sol.Result.At("x")[0]
	yhat := matrix.FloatVector(x.FloatArray()[:m])
	g := matrix.FloatVector(x.FloatArray()[m:])

	rangeFunc := func(n int)[]float64 {
		r := make([]float64, 0)
		for i := 0; i < n; i++ {
			r = append(r, float64(i)*2.2/float64(n))
		}
		return r
	}
	ts := rangeFunc(1000)
	fitFunc := func(points []float64)[]float64 {
		res := make([]float64, len(points))
		for k, t := range points {
			res[k] = matrix.Plus(yhat, matrix.Mul(g, matrix.Scale(u, -1.0).Add(t))).Max()
		}
		return res
	}
	fs := fitFunc(ts)
	plotData("cvxfit.png", u.FloatArray(), y.FloatArray(), ts, fs)
}


// Local Variables:
// tab-width: 4
// End:
