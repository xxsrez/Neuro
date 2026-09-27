package neuro

import (
	"math"
	"reflect"
	"testing"
)

func near(t *testing.T, actual, expected float64) {
	t.Helper()
	if !Finite(actual) || !Finite(expected) || math.Abs(actual-expected) > 1e-7+1e-4*math.Max(math.Abs(actual), math.Abs(expected)) {
		t.Fatalf("expected %.17g, got %.17g", expected, actual)
	}
}
func exactish(t *testing.T, actual, expected float64) {
	t.Helper()
	if math.Abs(actual-expected) > 1e-12 || !Finite(actual) {
		t.Fatalf("expected %.17g, got %.17g", expected, actual)
	}
}
func measure(t *testing.T, data []Sample, predict func([]float64) []float64) Metrics {
	t.Helper()
	m, err := Measure(data, predict)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func step(t *testing.T, o *Optimizer) {
	t.Helper()
	if err := o.Step(); err != nil {
		t.Fatal(err)
	}
}
func gradientCopy(n *Network) [][]float64 {
	result := make([][]float64, len(n.Parameters))
	for i, p := range n.Parameters {
		result[i] = append([]float64(nil), p.Gradients...)
	}
	return result
}
func orderOf(count int) []int {
	o := make([]int, count)
	for i := range o {
		o[i] = i
	}
	return o
}

func TestData(t *testing.T) {
	x, y := MotionStep(0.2, 0.7, 0.2)
	wantX := 0.2 + 0.2*math.Sin(0.7)
	near(t, x, wantX)
	near(t, y, 0.7-0.2*math.Sin(wantX))
	near(t, MotionEnd(2, -3, 0, 0.2)[0], 2)
	near(t, MotionEnd(2, -3, 20, 0)[1], -3)
	s := Point(2, -3, 0, 0.2)
	near(t, s.Input[0]*math.Pi, 2)
	near(t, s.Target[1]*math.Pi, -3)
	if !reflect.DeepEqual(Generate(16, 11, 20, 0.2), Generate(16, 11, 20, 0.2)) {
		t.Fatal("nonrepeatable generator")
	}
	if reflect.DeepEqual(Generate(16, 11, 20, 0.2), Generate(16, 12, 20, 0.2)) {
		t.Fatal("seeds not independent")
	}
	grid := Grid(3, 0, 0.2)
	near(t, grid[0].Input[0], -1)
	near(t, grid[8].Input[1], 1)
	near(t, grid[4].Input[0], 0)
	rng := RandomStream{}
	if rng.Next() != 0xe220a8397b1dcdaf {
		t.Fatal("SplitMix64 stream changed")
	}
}

func TestActivations(t *testing.T) {
	for _, a := range []Activation{Swish, ReLU, Tanh, LeakyReLU} {
		t.Run(string(a), func(t *testing.T) {
			for _, x := range []float64{-3, -0.2, 0.3, 2} {
				const eps = 1e-6
				near(t, a.Derivative(x), (a.Value(x+eps)-a.Value(x-eps))/(2*eps))
			}
		})
	}
	near(t, ReLU.Derivative(0), 0)
	near(t, LeakyReLU.Derivative(0), 0.01)
	for _, x := range []float64{-1000, 1000} {
		if !Finite(Swish.Value(x)) || !Finite(Swish.Derivative(x)) {
			t.Fatal("unstable Swish")
		}
	}
}

func TestDense(t *testing.T) {
	d := NewDense("d", 2, 2, &RandomStream{State: 1})
	copy(d.Weights.Values, []float64{1, 2, 3, 4})
	copy(d.Bias.Values, []float64{0.5, -0.5})
	x := []float64{2, -1}
	y := d.Forward(x)
	near(t, y[0], 0.5)
	near(t, y[1], 1.5)
	dx := d.Backward(x, []float64{0.1, 0.2})
	near(t, dx[0], 0.7)
	near(t, dx[1], 1)
	near(t, d.Weights.Gradients[0], 0.2)
	near(t, d.Weights.Gradients[3], -0.2)
	near(t, d.Bias.Gradients[1], 0.2)
}

func TestResidualIdentity(t *testing.T) {
	n := NewNetwork(20, 10, Swish, 0.1, 42)
	if n.ParameterCount() != 8502 {
		t.Fatal("wrong parameter count")
	}
	if &n.Blocks[0].First.Weights.Values[0] == &n.Blocks[1].First.Weights.Values[0] {
		t.Fatal("shared weights")
	}
	b := NewResidualBlock(0, 3, Swish, 0, &RandomStream{State: 42})
	input, g := []float64{0.2, -0.5, 0.7}, []float64{0.1, -0.3, 0.4}
	if !reflect.DeepEqual(b.Forward(input), input) || !reflect.DeepEqual(b.Backward(g), g) {
		t.Fatal("identity route failed")
	}
	for _, p := range []*Parameter{b.First.Weights, b.First.Bias, b.Second.Weights, b.Second.Bias} {
		for _, x := range p.Gradients {
			if x != 0 {
				t.Fatal("nonzero disabled branch gradient")
			}
		}
	}
}

func TestNetworkFiniteDifferences(t *testing.T) {
	for _, a := range []Activation{Swish, ReLU, Tanh, LeakyReLU} {
		t.Run(string(a), func(t *testing.T) {
			n := NewNetwork(3, 2, a, 0.1, 123)
			input, target := []float64{0.3, -0.4}, []float64{-0.2, 0.6}
			n.ZeroGradients()
			y := n.Forward(input)
			dx := append([]float64(nil), n.Backward(LossGradient(y, target, 1))...)
			gradients := gradientCopy(n)
			const eps = 1e-6
			for p, parameter := range n.Parameters {
				for i, original := range parameter.Values {
					parameter.Values[i] = original + eps
					plus := MSE(n.Forward(input), target)
					parameter.Values[i] = original - eps
					minus := MSE(n.Forward(input), target)
					parameter.Values[i] = original
					near(t, gradients[p][i], (plus-minus)/(2*eps))
				}
			}
			for i, original := range input {
				input[i] = original + eps
				plus := MSE(n.Forward(input), target)
				input[i] = original - eps
				minus := MSE(n.Forward(input), target)
				input[i] = original
				near(t, dx[i], (plus-minus)/(2*eps))
			}
		})
	}
}

func TestBatches(t *testing.T) {
	n := NewNetwork(3, 2, Swish, 0.1, 12)
	data := Generate(5, 33, 20, 0.2)
	order := []int{4, 2, 0, 1, 3}
	expected := gradientCopy(n)
	for j := 1; j < 4; j++ {
		s := data[order[j]]
		n.ZeroGradients()
		y := n.Forward(s.Input)
		n.Backward(LossGradient(y, s.Target, 1))
		for p, parameter := range n.Parameters {
			for i, g := range parameter.Gradients {
				expected[p][i] += g / 3
			}
		}
	}
	loss := Accumulate(n, data, order, 1, 3)
	actual := gradientCopy(n)
	for p := range expected {
		for i := range expected[p] {
			near(t, actual[p][i], expected[p][i])
		}
	}
	meanLoss := func() float64 {
		sum := 0.0
		for j := 1; j < 4; j++ {
			s := data[order[j]]
			sum += MSE(n.Forward(s.Input), s.Target) / 3
		}
		return sum
	}
	near(t, loss, meanLoss())
	const eps = 1e-6
	p := n.Parameters[0]
	original := p.Values[0]
	p.Values[0] = original + eps
	plus := meanLoss()
	p.Values[0] = original - eps
	minus := meanLoss()
	p.Values[0] = original
	near(t, actual[0][0], (plus-minus)/(2*eps))
	Accumulate(n, data, order, 4, 1)
	tail := gradientCopy(n)
	n.ZeroGradients()
	s := data[order[4]]
	y := n.Forward(s.Input)
	n.Backward(LossGradient(y, s.Target, 1))
	if !reflect.DeepEqual(tail, gradientCopy(n)) {
		t.Fatal("wrong tail scaling")
	}
}

func TestOptimizers(t *testing.T) {
	p := newParameter("p", 1)
	p.Values[0] = 1
	p.Gradients[0] = 2
	step(t, NewOptimizer([]*Parameter{p}, SGD, 0.1))
	exactish(t, p.Values[0], 0.8)
	p.Values[0] = 1
	adam := NewOptimizer([]*Parameter{p}, Adam, 0.1)
	step(t, adam)
	exactish(t, p.Values[0], 1-0.1*2/(2+1e-8))
	p.Gradients[0] = -1
	step(t, adam)
	m, v := 0.9*0.2-0.1, 0.999*0.004+0.001
	want := 1 - 0.1*2/(2+1e-8) - 0.1*(m/(1-0.81))/(math.Sqrt(v/(1-0.998001))+1e-8)
	exactish(t, p.Values[0], want)
	if adam.StepCount != 2 {
		t.Fatal("wrong step count")
	}
	p.Gradients[0] = math.Inf(1)
	if adam.Step() == nil {
		t.Fatal("accepted infinite gradient")
	}
}

func TestAffine(t *testing.T) {
	data := Generate(64, 2, 0, 0.2)
	for i, s := range data {
		data[i].Target = []float64{2*s.Input[0] - s.Input[1] + 0.3, s.Input[0] + 3*s.Input[1] - 0.4}
	}
	a, err := FitAffine(data)
	if err != nil {
		t.Fatal(err)
	}
	if measure(t, data, a.Predict).MSE > 1e-25 {
		t.Fatal("affine fit incorrect")
	}
	if _, err := FitAffine([]Sample{{[]float64{0, 0}, []float64{1, 1}}}); err == nil {
		t.Fatal("accepted singular data")
	}
}

func TestReproducibility(t *testing.T) {
	data := Generate(17, 2, 20, 0.2)
	a, b := NewNetwork(4, 2, Swish, 0.1, 4), NewNetwork(4, 2, Swish, 0.1, 4)
	saved, gradients := a.Snapshot(), gradientCopy(a)
	measure(t, data, a.Forward)
	if !reflect.DeepEqual(saved, a.Snapshot()) || !reflect.DeepEqual(gradients, gradientCopy(a)) {
		t.Fatal("evaluation mutates weights/gradients")
	}
	oa, ob := NewOptimizer(a.Parameters, Adam, 0.001), NewOptimizer(b.Parameters, Adam, 0.001)
	order := orderOf(len(data))
	for i := 0; i < 3; i++ {
		Accumulate(a, data, order, 0, 17)
		step(t, oa)
		Accumulate(b, data, order, 0, 17)
		step(t, ob)
	}
	if !reflect.DeepEqual(a.Snapshot(), b.Snapshot()) {
		t.Fatal("training not repeatable")
	}
	n := NewNetwork(4, 2, Swish, 0.1, 5)
	if err := n.Restore(a.Snapshot()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Forward(data[0].Input), n.Forward(data[0].Input)) {
		t.Fatal("restored prediction differs")
	}
	bad := a.Snapshot()
	bad[len(bad)-1] = []float64{1}
	if err := a.Restore(bad); err == nil {
		t.Fatal("accepted malformed checkpoint")
	}
	if !reflect.DeepEqual(a.Snapshot(), b.Snapshot()) {
		t.Fatal("invalid restore partially mutated model")
	}
}

func TestLearning(t *testing.T) {
	data := Generate(64, 123, 0, 0.2)
	n := NewNetwork(4, 2, Swish, 0.1, 4)
	o := NewOptimizer(n.Parameters, Adam, 0.01)
	order := orderOf(len(data))
	before := measure(t, data, n.Forward).MSE
	for i := 0; i < 300; i++ {
		Accumulate(n, data, order, 0, len(data))
		step(t, o)
	}
	after := measure(t, Generate(64, 321, 0, 0.2), n.Forward).MSE
	if after >= 0.001 || after >= before*0.01 {
		t.Fatalf("did not learn: %g -> %g", before, after)
	}
}
