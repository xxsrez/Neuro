package app

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"neuro/internal/neuro"
)

type Config struct {
	Width           int                 `json:"width"`
	Blocks          int                 `json:"blocks"`
	Activation      neuro.Activation    `json:"activation"`
	Optimizer       neuro.OptimizerKind `json:"optimizer"`
	Alpha           float64             `json:"alpha"`
	LearningRate    float64             `json:"learningRate"`
	Epochs          int                 `json:"epochs"`
	BatchSize       int                 `json:"batchSize"`
	TrainCount      int                 `json:"trainCount"`
	ValidationCount int                 `json:"validationCount"`
	TestCount       int                 `json:"testCount"`
	Steps           int                 `json:"steps"`
	H               float64             `json:"h"`
	GridSide        int                 `json:"gridSide"`
	TrainSeed       uint64              `json:"trainSeed"`
	ValidationSeed  uint64              `json:"validationSeed"`
	TestSeed        uint64              `json:"testSeed"`
	WeightSeed      uint64              `json:"weightSeed"`
	ShuffleSeed     uint64              `json:"shuffleSeed"`
	Output          string              `json:"output"`
	AcceptanceMSE   float64             `json:"acceptanceMse"`
}

func DefaultConfig() Config {
	return Config{20, 10, neuro.Swish, neuro.Adam, 0.1, 0.001, 200, 64, 8192, 2048, 2048, 20, 0.2, 101, 1001, 2002, 3003, 42, 4004, "runs/go-prototype", 0.01}
}

func ParseConfig(args []string) (Config, error) {
	c := DefaultConfig()
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.IntVar(&c.Width, "width", c.Width, "")
	fs.IntVar(&c.Blocks, "blocks", c.Blocks, "")
	a, o := string(c.Activation), string(c.Optimizer)
	fs.StringVar(&a, "activation", a, "")
	fs.StringVar(&o, "optimizer", o, "")
	fs.Float64Var(&c.Alpha, "alpha", c.Alpha, "")
	fs.Float64Var(&c.LearningRate, "lr", c.LearningRate, "")
	fs.IntVar(&c.Epochs, "epochs", c.Epochs, "")
	fs.IntVar(&c.BatchSize, "batch", c.BatchSize, "")
	fs.IntVar(&c.TrainCount, "train", c.TrainCount, "")
	fs.IntVar(&c.ValidationCount, "validation", c.ValidationCount, "")
	fs.IntVar(&c.TestCount, "test", c.TestCount, "")
	fs.IntVar(&c.Steps, "steps", c.Steps, "")
	fs.Float64Var(&c.H, "h", c.H, "")
	fs.IntVar(&c.GridSide, "grid", c.GridSide, "")
	fs.Uint64Var(&c.TrainSeed, "train-seed", c.TrainSeed, "")
	fs.Uint64Var(&c.ValidationSeed, "validation-seed", c.ValidationSeed, "")
	fs.Uint64Var(&c.TestSeed, "test-seed", c.TestSeed, "")
	fs.Uint64Var(&c.WeightSeed, "weight-seed", c.WeightSeed, "")
	fs.Uint64Var(&c.ShuffleSeed, "shuffle-seed", c.ShuffleSeed, "")
	fs.StringVar(&c.Output, "out", c.Output, "")
	if err := fs.Parse(args); err != nil {
		return c, err
	}
	if fs.NArg() != 0 {
		return c, fmt.Errorf("unexpected argument: %s", fs.Arg(0))
	}
	for _, kind := range []neuro.Activation{neuro.Swish, neuro.ReLU, neuro.Tanh, neuro.LeakyReLU} {
		if strings.EqualFold(a, string(kind)) {
			a = string(kind)
		}
	}
	for _, kind := range []neuro.OptimizerKind{neuro.Adam, neuro.SGD} {
		if strings.EqualFold(o, string(kind)) {
			o = string(kind)
		}
	}
	c.Activation = neuro.Activation(a)
	c.Optimizer = neuro.OptimizerKind(o)
	return c, c.Validate()
}

func (c Config) Validate() error {
	if c.Width < 1 || c.Width > 256 || c.Blocks < 0 || c.Blocks > 100 || c.Epochs < 1 || c.Epochs > 10000 || c.BatchSize < 1 ||
		c.TrainCount < 4 || c.TrainCount > 1000000 || c.ValidationCount < 1 || c.ValidationCount > 1000000 || c.TestCount < 1 || c.TestCount > 1000000 ||
		c.Steps < 0 || c.Steps > 10000 || c.GridSide < 2 || c.GridSide > 201 {
		return fmt.Errorf("invalid or excessive dimensions; see --help")
	}
	if !c.Activation.Valid() || (c.Optimizer != neuro.Adam && c.Optimizer != neuro.SGD) || !neuro.Finite(c.Alpha) ||
		!neuro.Finite(c.H) || c.H < 0 || !neuro.Finite(c.LearningRate) || c.LearningRate <= 0 || !neuro.Finite(c.AcceptanceMSE) || c.AcceptanceMSE < 0 {
		return fmt.Errorf("invalid activation, optimizer or numeric setting")
	}
	if c.TrainSeed == c.ValidationSeed || c.TrainSeed == c.TestSeed || c.ValidationSeed == c.TestSeed {
		return fmt.Errorf("use distinct dataset seeds")
	}
	if strings.TrimSpace(c.Output) == "" {
		return fmt.Errorf("empty output directory")
	}
	return nil
}
