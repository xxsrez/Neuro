package neuro

import "math"

type Activation string

const (
	Swish     Activation = "Swish"
	ReLU      Activation = "Relu"
	Tanh      Activation = "Tanh"
	LeakyReLU Activation = "LeakyRelu"
)

func (a Activation) Valid() bool { return a == Swish || a == ReLU || a == Tanh || a == LeakyReLU }

func Sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	t := math.Exp(x)
	return t / (1 + t)
}

func (a Activation) Value(x float64) float64 {
	switch a {
	case Swish:
		return x * Sigmoid(x)
	case ReLU:
		return math.Max(0, x)
	case Tanh:
		return math.Tanh(x)
	case LeakyReLU:
		if x >= 0 {
			return x
		}
		return 0.01 * x
	default:
		panic("invalid activation")
	}
}

func (a Activation) Derivative(x float64) float64 {
	switch a {
	case Swish:
		s := Sigmoid(x)
		return s + x*s*(1-s)
	case ReLU:
		if x > 0 {
			return 1
		}
		return 0
	case Tanh:
		t := math.Tanh(x)
		return 1 - t*t
	case LeakyReLU:
		if x > 0 {
			return 1
		}
		return 0.01
	default:
		panic("invalid activation")
	}
}
