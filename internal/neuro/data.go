package neuro

import "math"

const MotionVersion = "sequential-sine-v1"

// SplitMix64 is shared with the original prototype. Use separate instances
// for data, weights and shuffling so changing one stream cannot alter another.
type RandomStream struct{ State uint64 }

func (r *RandomStream) Next() uint64 {
	r.State += 0x9E3779B97F4A7C15
	z := r.State
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *RandomStream) Uniform() float64 { return float64(r.Next()>>11) * (1.0 / (1 << 53)) }

func (r *RandomStream) Shuffle(order []int) {
	for i := len(order) - 1; i > 0; i-- {
		j := int(r.Uniform() * float64(i+1))
		order[i], order[j] = order[j], order[i]
	}
}

type Sample struct{ Input, Target []float64 }

func MotionStep(x, y, h float64) (float64, float64) {
	nextX := x + h*math.Sin(y)
	return nextX, y - h*math.Sin(nextX)
}

func MotionEnd(x, y float64, steps int, h float64) []float64 {
	if steps < 0 || !Finite(h) {
		panic("invalid motion settings")
	}
	for i := 0; i < steps; i++ {
		x, y = MotionStep(x, y, h)
	}
	return []float64{x, y}
}

func Point(x, y float64, steps int, h float64) Sample {
	end := MotionEnd(x, y, steps, h)
	return Sample{[]float64{x / math.Pi, y / math.Pi}, []float64{end[0] / math.Pi, end[1] / math.Pi}}
}

func Generate(count int, seed uint64, steps int, h float64) []Sample {
	if count < 1 {
		panic("empty dataset")
	}
	rng := RandomStream{State: seed}
	data := make([]Sample, count)
	for i := range data {
		data[i] = Point((2*rng.Uniform()-1)*math.Pi, (2*rng.Uniform()-1)*math.Pi, steps, h)
	}
	return data
}

// Rows run from -pi to +pi; visualizations invert the screen y axis.
func Grid(side, steps int, h float64) []Sample {
	if side < 2 {
		panic("grid side must be >= 2")
	}
	data := make([]Sample, side*side)
	for i := range data {
		x := math.Pi * (2*float64(i%side)/float64(side-1) - 1)
		y := math.Pi * (2*float64(i/side)/float64(side-1) - 1)
		data[i] = Point(x, y, steps, h)
	}
	return data
}

func Finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
