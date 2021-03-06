// This file is part of go.opt package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package main

import (
    "fmt"
    "github.com/ajstarks/svgo"
    "github.com/hrautila/cvx"
    "github.com/hrautila/cvx/sets"
    "github.com/hrautila/matrix"
    "os"
)

func drawPlan(name string, xs, ys, ws, hs []float64) {
    fill := []string{
        "fill:gray;stroke:black",
        "fill:red;stroke:black",
        "fill:blue;stroke:black",
        "fill:green;stroke:black",
        "fill:yellow;stroke:black"}

    f, err := os.Create(name)
    if err != nil {
        fmt.Printf("create error: %v\n", err)
        return
    }

    g := svg.New(f)
    g.Start(260, 260)
    for k, _ := range xs {
        g.Rect(int(10.0*xs[k])+2, int(10.0*ys[k])+5, int(10.*ws[k]), int(10.0*hs[k]), fill[k])
    }
    g.End()
}

// floorPlan implements interface cvx.ConvexProg.
type floorPlan struct {
    Amin *matrix.FloatMatrix
}

func newFloorPlan(plan *matrix.FloatMatrix) *floorPlan {
    return &floorPlan{plan}
}

func (p *floorPlan) F0() (mnl int, x0 *matrix.FloatMatrix, err error) {
    err = nil
    mnl = 5
    x0 = matrix.FloatZeros(22, 1)
    // set last 5 elements to 1.0
    x0.SetIndexes(1.0, -1, -2, -3, -4, -5)
    return
}

func (p *floorPlan) F1(x *matrix.FloatMatrix) (f, Df *matrix.FloatMatrix, err error) {
    err = nil
    mn := x.Min(-1, -2, -3, -4, -5)
    if mn <= 0.0 {
        f, Df = nil, nil
        return
    }
    zeros := matrix.FloatZeros(5, 12)
    dk1 := matrix.FloatDiagonal(5, -1.0)
    dk2 := matrix.FloatZeros(5, 5)
    //x17 := matrix.FloatVector(x.FloatArray()[17:])
	x17 := x.SubMatrix(17, 0).Copy()
    // -( Amin ./ (x17 .* x17) )
    diag := matrix.Div(p.Amin, matrix.Mul(x17, x17)).Scale(-1.0)
    //dk2.SetIndexesFromArray(diag.FloatArray(), matrix.MakeDiagonalSet(5)...)
	diag.CopyTo(dk2.Diag())
    Df, _ = matrix.FloatMatrixStacked(matrix.StackRight, zeros, dk1, dk2)

    //x12 := matrix.FloatVector(x.FloatArray()[12:17])
	x12 := x.SubMatrix(12, 0, 5, 1)
    // f = -x[12:17] + div(Amin, x[17:]) == div(Amin, x[17:]) - x[12:17]
    f = matrix.Minus(matrix.Div(p.Amin, x17), x12)
    return
}

func (p *floorPlan) F2(x, z *matrix.FloatMatrix) (f, Df, H *matrix.FloatMatrix, err error) {
    f, Df, err = p.F1(x)
    //x17 := matrix.FloatVector(x.FloatArray()[17:])
	x17 := x.SubMatrix(17, 0).Copy()
    tmp := matrix.Div(p.Amin, matrix.Pow(x17, 3.0))
    tmp = matrix.Mul(z, tmp).Scale(2.0)
    diag := matrix.FloatDiagonal(5, tmp.FloatArray()...)
    H = matrix.FloatZeros(22, 22)
	H.SubMatrix(17, 17).Set(diag)
    //H.SetSubMatrix(17, 17, diag)
    return
}

// helpers
func row(m *matrix.FloatMatrix, row int) *matrix.FloatMatrix {
	return m.SubMatrix(row, 0, 1, m.Cols())
}

func floorplan(Amin *matrix.FloatMatrix) (W, H float64, xs, ys, ws, hs []float64, err error) {
    err = nil
    W, H = 0.0, 0.0
    xs, ys, ws, hs = nil, nil, nil, nil

    rho := 1.0
    gamma := 5.0

    c := matrix.FloatZeros(22, 1)
    c.SetAtColumnArray(0, []int{0, 1}, []float64{1.0, 1.0})

    G := matrix.FloatZeros(26, 22)
    h := matrix.FloatZeros(26, 1)

    // -x1 <= 0  
    G.SetAt(0, 2, -1.0)

    // -x2 <= 0   
    G.SetAt(1, 3, -1.0)

    // -x4 <= 0  
    G.SetAt(2, 5, -1.0)

    // x1 - x3 + w1 <= -rho
    //G.SetAtRowArray(3, []int{2, 4, 12}, []float64{1.0, -1.0, 1.0})
	row(G, 3).SetIndexesFromArray([]float64{1.0, -1.0, 1.0}, 2, 4, 12)
    h.SetAt(3, 0, -rho)

    // x2 - x3 + w2 <= -rho
    //G.SetAtRowArray(4, []int{3, 4, 13}, []float64{1.0, -1.0, 1.0})
	row(G, 4).SetIndexesFromArray([]float64{1.0, -1.0, 1.0}, 3, 4, 13)
    h.SetAt(4, 0, -rho)

    // x3 - x5 + w3 <= -rho
    //G.SetAtRowArray(5, []int{4, 6, 14}, []float64{1.0, -1.0, 1.0})
	row(G, 5).SetIndexesFromArray([]float64{1.0, -1.0, 1.0}, 4, 6, 10)
    h.SetAt(5, 0, -rho)

    // x4 - x5 + w4 <= -rho
    //G.SetAtRowArray(6, []int{5, 6, 15}, []float64{1.0, -1.0, 1.0})
	row(G, 6).SetIndexesFromArray([]float64{1.0, -1.0, 1.0}, 5, 6, 15)
    h.SetAt(6, 0, -rho)

    // -W + x5 + w5 <= 0  
    //G.SetAtRowArray(7, []int{0, 6, 16}, []float64{-1.0, 1.0, 1.0})
	row(G, 7).SetIndexesFromArray([]float64{-1.0, 1.0, 1.0}, 0, 6, 16)

    // -y2 <= 0  
    G.SetAt(8, 8, -1.0)

    // -y3 <= 0  
    G.SetAt(9, 9, -1.0)

    // -y5 <= 0  
    G.SetAt(10, 11, -1.0)

    // -y1 + y2 + h2 <= -rho  
    //G.SetAtRowArray(11, []int{7, 8, 18}, []float64{-1.0, 1.0, 1.0})
	row(G, 11).SetIndexesFromArray([]float64{-1.0, 1.0, 1.0}, 7, 8, 18)
    h.SetAt(11, 0, -rho)

    // y1 - y4 + h1 <= -rho  
    //G.SetAtRowArray(12, []int{7, 10, 17}, []float64{1.0, -1.0, 1.0})
	row(G, 12).SetIndexesFromArray([]float64{1.0, -1.0, 1.0}, 7, 10, 17)
    h.SetAt(12, 0, -rho)

    // y3 - y4 + h3 <= -rho  
    //G.SetAtRowArray(13, []int{9, 10, 19}, []float64{1.0, -1.0, 1.0})
	row(G, 13).SetIndexesFromArray([]float64{1.0, -1.0, 1.0}, 9, 10, 19)
    h.SetAt(13, 0, -rho)

    // -H + y4 + h4 <= 0  
    //G.SetAtRowArray(14, []int{1, 10, 20}, []float64{-1.0, 1.0, 1.0})
	row(G, 14).SetIndexesFromArray([]float64{-1.0, 1.0, 1.0}, 1, 10, 20)

    // -H + y5 + h5 <= 0  
    //G.SetAtRowArray(15, []int{1, 11, 21}, []float64{-1.0, 1.0, 1.0})
	row(G, 15).SetIndexesFromArray([]float64{-1.0, 1.0, 1.0}, 1, 11, 21)

    // -w1 + h1/gamma <= 0  
    //G.SetAtRowArray(16, []int{12, 17}, []float64{-1.0, 1.0 / gamma})
	row(G, 16).SetIndexesFromArray([]float64{-1.0, 1.0 / gamma}, 12, 17)

    // w1 - gamma * h1 <= 0  
    //G.SetAtRowArray(17, []int{12, 17}, []float64{1.0, -gamma})
	row(G, 17).SetIndexesFromArray([]float64{1.0, -gamma}, 12, 17)

    // -w2 + h2/gamma <= 0  
    //G.SetAtRowArray(18, []int{13, 18}, []float64{-1.0, 1.0 / gamma})
	row(G, 18).SetIndexesFromArray([]float64{-1.0, 1.0 / gamma}, 13, 18)

    //  w2 - gamma * h2 <= 0  
    //G.SetAtRowArray(19, []int{13, 18}, []float64{1.0, -gamma})
	row(G, 19).SetIndexesFromArray([]float64{1.0, -gamma}, 13, 18)

    // -w3 + h3/gamma <= 0  
    //G.SetAtRowArray(20, []int{14, 18}, []float64{-1.0, 1.0 / gamma})
	row(G, 20).SetIndexesFromArray([]float64{-1.0, 1.0 / gamma}, 14, 18)

    //  w3 - gamma * h3 <= 0  
    //G.SetAtRowArray(21, []int{14, 19}, []float64{1.0, -gamma})
	row(G, 21).SetIndexesFromArray([]float64{1.0, -gamma}, 14, 19)

    // -w4  + h4/gamma <= 0  
    //G.SetAtRowArray(22, []int{15, 19}, []float64{-1.0, 1.0 / gamma})
	row(G, 22).SetIndexesFromArray([]float64{-1.0, 1.0 / gamma}, 15, 19)

    //  w4 - gamma * h4 <= 0  
    //G.SetAtRowArray(23, []int{15, 20}, []float64{1.0, -gamma})
	row(G, 23).SetIndexesFromArray([]float64{1.0, -gamma}, 15, 20)

    // -w5 + h5/gamma <= 0  
    //G.SetAtRowArray(24, []int{16, 21}, []float64{-1.0, 1.0 / gamma})
	row(G, 24).SetIndexesFromArray([]float64{-1.0, 1.0 / gamma}, 16, 21)

    //  w5 - gamma * h5 <= 0.0  
    //G.SetAtRowArray(25, []int{16, 21}, []float64{1.0, -gamma})
	row(G, 25).SetIndexesFromArray([]float64{1.0, -gamma}, 16, 21)

    F := newFloorPlan(Amin)

    var dims *sets.DimensionSet = nil
    var solopts cvx.SolverOptions
    var sol *cvx.Solution
    solopts.MaxIter = 50
    solopts.ShowProgress = false
    sol, err = cvx.Cpl(F, c, G, h, nil, nil, dims, &solopts)
    if err == nil && sol.Status == cvx.Optimal {
        x := sol.Result.At("x")[0]
        W = x.GetIndex(0)
        H = x.GetIndex(1)
        xs = x.FloatArray()[2:7]
        ys = x.FloatArray()[7:12]
        ws = x.FloatArray()[12:17]
        hs = x.FloatArray()[17:]
        return
    } else {
        fmt.Printf("result: %v\n", err)
    }
    return
}

func main() {

    dataA := []float64{100.0, 100.0, 100.0, 100.0, 100.0}
    dataB := []float64{20.0, 50.0, 80.0, 150.0, 200.0}
    dataC := []float64{180.0, 80.0, 80.0, 80.0, 80.0}
    dataD := []float64{20.0, 150.0, 20.0, 200.0, 110.0}

    W, H, xs, ys, ws, hs, err := floorplan(matrix.FloatVector(dataA))
    fmt.Printf("plan A: W=%.2f, H=%.2f\n", W, H)
    if err == nil {
        drawPlan("planA.svg", xs, ys, ws, hs)
    }
    W, H, xs, ys, ws, hs, err = floorplan(matrix.FloatVector(dataB))
    fmt.Printf("plan B: W=%.2f, H=%.2f\n", W, H)
    if err == nil {
        drawPlan("planB.svg", xs, ys, ws, hs)
    }
    W, H, xs, ys, ws, hs, err = floorplan(matrix.FloatVector(dataC))
    fmt.Printf("plan C: W=%.2f, H=%.2f\n", W, H)
    if err == nil {
        drawPlan("planC.svg", xs, ys, ws, hs)
    }
    W, H, xs, ys, ws, hs, err = floorplan(matrix.FloatVector(dataD))
    fmt.Printf("plan D: W=%.2f, H=%.2f\n", W, H)
    if err == nil {
        drawPlan("planD.svg", xs, ys, ws, hs)
    }
}

// Local Variables:
// tab-width: 4
// End:
