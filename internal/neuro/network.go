package neuro

import (
	"fmt"
	"math"
)

type Parameter struct {
	Name              string
	Values, Gradients []float64
}

func newParameter(name string, count int) *Parameter {
	return &Parameter{name, make([]float64, count), make([]float64, count)}
}

// Row-major W[output*inputSize+input]. Buffers are reused: the model is
// single-threaded, and Backward must follow the matching Forward.
type Dense struct {
	InputSize, OutputSize int
	Weights, Bias         *Parameter
	Output, inputGradient []float64
}

func NewDense(name string, input, output int, rng *RandomStream) *Dense {
	if input < 1 || output < 1 {
		panic("invalid layer dimensions")
	}
	d := &Dense{InputSize: input, OutputSize: output, Weights: newParameter(name+".weight", input*output),
		Bias: newParameter(name+".bias", output), Output: make([]float64, output), inputGradient: make([]float64, input)}
	limit := math.Sqrt(6.0 / float64(input+output))
	for i := range d.Weights.Values {
		d.Weights.Values[i] = (2*rng.Uniform() - 1) * limit
	}
	return d
}

func (d *Dense) Forward(input []float64) []float64 {
	if len(input) != d.InputSize {
		panic("wrong input dimension")
	}
	for o := 0; o < d.OutputSize; o++ {
		sum := d.Bias.Values[o]
		offset := o * d.InputSize
		for i := 0; i < d.InputSize; i++ {
			sum += d.Weights.Values[offset+i] * input[i]
		}
		d.Output[o] = sum
	}
	return d.Output
}

func (d *Dense) Backward(input, gradient []float64) []float64 {
	if len(input) != d.InputSize || len(gradient) != d.OutputSize {
		panic("wrong gradient dimension")
	}
	clear(d.inputGradient)
	for o := 0; o < d.OutputSize; o++ {
		g := gradient[o]
		offset := o * d.InputSize
		d.Bias.Gradients[o] += g
		for i := 0; i < d.InputSize; i++ {
			// dW = g outer input, dx = transpose(W) * g.
			d.Weights.Gradients[offset+i] += g * input[i]
			d.inputGradient[i] += g * d.Weights.Values[offset+i]
		}
	}
	return d.inputGradient
}

type ResidualBlock struct {
	First, Second                                           *Dense
	Activated, Output                                       []float64
	LastInputGradientRMS                                    float64
	activation                                              Activation
	alpha                                                   float64
	branchGradient, preGradient, inputGradient, cachedInput []float64
}

func NewResidualBlock(index, width int, a Activation, alpha float64, rng *RandomStream) *ResidualBlock {
	return &ResidualBlock{
		First:     NewDense(fmt.Sprintf("block%d.first", index), width, width, rng),
		Second:    NewDense(fmt.Sprintf("block%d.second", index), width, width, rng),
		Activated: make([]float64, width), Output: make([]float64, width), activation: a, alpha: alpha,
		branchGradient: make([]float64, width), preGradient: make([]float64, width), inputGradient: make([]float64, width),
	}
}

func (b *ResidualBlock) Forward(input []float64) []float64 {
	b.cachedInput = input
	a := b.First.Forward(input)
	for i, x := range a {
		b.Activated[i] = b.activation.Value(x)
	}
	r := b.Second.Forward(b.Activated)
	for i, x := range input {
		b.Output[i] = x + b.alpha*r[i]
	}
	return b.Output
}

func (b *ResidualBlock) Backward(gradient []float64) []float64 {
	if b.cachedInput == nil {
		panic("forward required before backward")
	}
	for i, g := range gradient {
		b.branchGradient[i] = b.alpha * g
	}
	qGradient := b.Second.Backward(b.Activated, b.branchGradient)
	for i, g := range qGradient {
		b.preGradient[i] = g * b.activation.Derivative(b.First.Output[i])
	}
	branchInput := b.First.Backward(b.cachedInput, b.preGradient)
	squares := 0.0
	for i, g := range gradient {
		b.inputGradient[i] = g + branchInput[i] // Identity route.
		squares += b.inputGradient[i] * b.inputGradient[i]
	}
	b.LastInputGradientRMS = math.Sqrt(squares / float64(len(gradient)))
	return b.inputGradient
}

type Network struct {
	Input, Output *Dense
	Blocks        []*ResidualBlock
	Parameters    []*Parameter
	cachedInput   []float64
}

func NewNetwork(width, blocks int, a Activation, alpha float64, seed uint64) *Network {
	if width < 1 || blocks < 0 || !a.Valid() || !Finite(alpha) {
		panic("invalid network settings")
	}
	rng := &RandomStream{State: seed}
	n := &Network{Input: NewDense("input", 2, width, rng), Blocks: make([]*ResidualBlock, blocks)}
	n.Parameters = []*Parameter{n.Input.Weights, n.Input.Bias}
	for i := range n.Blocks {
		b := NewResidualBlock(i, width, a, alpha, rng)
		n.Blocks[i] = b
		n.Parameters = append(n.Parameters, b.First.Weights, b.First.Bias, b.Second.Weights, b.Second.Bias)
	}
	n.Output = NewDense("output", width, 2, rng)
	n.Parameters = append(n.Parameters, n.Output.Weights, n.Output.Bias)
	return n
}

func (n *Network) ParameterCount() int {
	count := 0
	for _, p := range n.Parameters {
		count += len(p.Values)
	}
	return count
}

// Forward returns a reused buffer. Copy it if the result must survive another call.
func (n *Network) Forward(input []float64) []float64 {
	n.cachedInput = input
	z := n.Input.Forward(input)
	for _, b := range n.Blocks {
		z = b.Forward(z)
	}
	return n.Output.Forward(z)
}

func (n *Network) Backward(gradient []float64) []float64 {
	if n.cachedInput == nil {
		panic("forward required before backward")
	}
	z := n.Input.Output
	if len(n.Blocks) > 0 {
		z = n.Blocks[len(n.Blocks)-1].Output
	}
	g := n.Output.Backward(z, gradient)
	for i := len(n.Blocks) - 1; i >= 0; i-- {
		g = n.Blocks[i].Backward(g)
	}
	return n.Input.Backward(n.cachedInput, g)
}

func (n *Network) ZeroGradients() {
	for _, p := range n.Parameters {
		clear(p.Gradients)
	}
}

func (n *Network) Snapshot() [][]float64 {
	values := make([][]float64, len(n.Parameters))
	for i, p := range n.Parameters {
		values[i] = append([]float64(nil), p.Values...)
	}
	return values
}

func (n *Network) Restore(values [][]float64) error {
	if len(values) != len(n.Parameters) {
		return fmt.Errorf("wrong parameter count")
	}
	for i, v := range values {
		if len(v) != len(n.Parameters[i].Values) {
			return fmt.Errorf("wrong parameter shape at %d", i)
		}
		for _, x := range v {
			if !Finite(x) {
				return fmt.Errorf("non-finite weight")
			}
		}
	}
	for i, v := range values {
		copy(n.Parameters[i].Values, v)
	}
	return nil
}

func MSE(prediction, target []float64) float64 {
	if len(prediction) != 2 || len(target) != 2 {
		panic("expected two coordinates")
	}
	x, y := prediction[0]-target[0], prediction[1]-target[1]
	return (x*x + y*y) / 2
}

func LossGradient(prediction, target []float64, batchSize int) []float64 {
	if batchSize < 1 {
		panic("invalid batch size")
	}
	return []float64{(prediction[0] - target[0]) / float64(batchSize), (prediction[1] - target[1]) / float64(batchSize)}
}
