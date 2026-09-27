package neuro

import (
	"fmt"
	"math"
)

type OptimizerKind string

const (
	Adam OptimizerKind = "Adam"
	SGD  OptimizerKind = "Sgd"
)

type Optimizer struct {
	Parameters    []*Parameter
	StepCount     int64
	kind          OptimizerKind
	lr            float64
	first, second [][]float64
}

func NewOptimizer(parameters []*Parameter, kind OptimizerKind, lr float64) *Optimizer {
	if !Finite(lr) || lr <= 0 || (kind != Adam && kind != SGD) {
		panic("invalid optimizer settings")
	}
	o := &Optimizer{Parameters: parameters, kind: kind, lr: lr, first: make([][]float64, len(parameters)), second: make([][]float64, len(parameters))}
	for i, p := range parameters {
		o.first[i] = make([]float64, len(p.Values))
		o.second[i] = make([]float64, len(p.Values))
	}
	return o
}

func (o *Optimizer) Step() error {
	o.StepCount++
	c1, c2 := 1-math.Pow(0.9, float64(o.StepCount)), 1-math.Pow(0.999, float64(o.StepCount))
	for p, parameter := range o.Parameters {
		for i, g := range parameter.Gradients {
			if !Finite(g) {
				return fmt.Errorf("non-finite gradient %s[%d]", parameter.Name, i)
			}
			update := o.lr * g
			if o.kind == Adam {
				o.first[p][i] = 0.9*o.first[p][i] + 0.1*g
				o.second[p][i] = 0.999*o.second[p][i] + 0.001*g*g
				if !Finite(o.first[p][i]) || !Finite(o.second[p][i]) {
					return fmt.Errorf("non-finite Adam state")
				}
				update = o.lr * (o.first[p][i] / c1) / (math.Sqrt(o.second[p][i]/c2) + 1e-8)
			}
			parameter.Values[i] -= update
			if !Finite(parameter.Values[i]) {
				return fmt.Errorf("non-finite parameter after update")
			}
		}
	}
	return nil
}

type Metrics struct {
	MSE          float64 `json:"mse"`
	MeanDistance float64 `json:"meanDistance"`
}

func Measure(data []Sample, predict func([]float64) []float64) (Metrics, error) {
	if len(data) == 0 {
		return Metrics{}, fmt.Errorf("empty evaluation dataset")
	}
	var m Metrics
	for _, sample := range data {
		loss := MSE(predict(sample.Input), sample.Target)
		if !Finite(loss) {
			return Metrics{}, fmt.Errorf("non-finite evaluation")
		}
		m.MSE += loss
		m.MeanDistance += math.Sqrt(2*loss) * math.Pi
	}
	m.MSE /= float64(len(data))
	m.MeanDistance /= float64(len(data))
	return m, nil
}

func Accumulate(n *Network, data []Sample, order []int, start, count int) float64 {
	if count < 1 || start < 0 || start+count > len(order) {
		panic("invalid batch range")
	}
	n.ZeroGradients()
	loss := 0.0
	for j := start; j < start+count; j++ {
		s := data[order[j]]
		output := n.Forward(s.Input)
		loss += MSE(output, s.Target)
		n.Backward(LossGradient(output, s.Target, count))
	}
	return loss / float64(count)
}

type AffineBaseline struct{ Coefficients [][]float64 }

// Fit y = Ax + b on training data only. The normal equation has just three
// features (x,y,1); solve it with our own pivoted Gaussian elimination.
func FitAffine(data []Sample) (*AffineBaseline, error) {
	var gram [3][3]float64
	var rhs [3][2]float64
	for _, s := range data {
		f := [3]float64{s.Input[0], s.Input[1], 1}
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				gram[r][c] += f[r] * f[c]
			}
			for c := 0; c < 2; c++ {
				rhs[r][c] += f[r] * s.Target[c]
			}
		}
	}
	var a [3][5]float64
	for r := 0; r < 3; r++ {
		copy(a[r][:3], gram[r][:])
		copy(a[r][3:], rhs[r][:])
	}
	for k := 0; k < 3; k++ {
		pivot := k
		for r := k + 1; r < 3; r++ {
			if math.Abs(a[r][k]) > math.Abs(a[pivot][k]) {
				pivot = r
			}
		}
		if math.Abs(a[pivot][k]) < 1e-12 {
			return nil, fmt.Errorf("singular affine training data")
		}
		a[k], a[pivot] = a[pivot], a[k]
		divisor := a[k][k]
		for c := k; c < 5; c++ {
			a[k][c] /= divisor
		}
		for r := 0; r < 3; r++ {
			if r == k {
				continue
			}
			factor := a[r][k]
			for c := k; c < 5; c++ {
				a[r][c] -= factor * a[k][c]
			}
		}
	}
	return &AffineBaseline{[][]float64{{a[0][3], a[1][3], a[2][3]}, {a[0][4], a[1][4], a[2][4]}}}, nil
}

func (a *AffineBaseline) Predict(input []float64) []float64 {
	output := make([]float64, 2)
	for i, c := range a.Coefficients {
		output[i] = c[0]*input[0] + c[1]*input[1] + c[2]
	}
	return output
}
