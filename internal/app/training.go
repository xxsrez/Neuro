package app

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"neuro/internal/neuro"
)

type EpochRecord struct {
	Epoch             int           `json:"epoch"`
	Train             neuro.Metrics `json:"train"`
	Validation        neuro.Metrics `json:"validation"`
	TrainSeconds      float64       `json:"trainSeconds"`
	EvaluationSeconds float64       `json:"evaluationSeconds"`
}
type WeightFile struct {
	FormatVersion int         `json:"formatVersion"`
	Configuration Config      `json:"configuration"`
	Epoch         int         `json:"epoch"`
	Parameters    [][]float64 `json:"parameters"`
}
type PointView struct {
	X          float64 `json:"x"`
	Y          float64 `json:"y"`
	TargetX    float64 `json:"targetX"`
	TargetY    float64 `json:"targetY"`
	PredictedX float64 `json:"predictedX"`
	PredictedY float64 `json:"predictedY"`
	Error      float64 `json:"error"`
}
type GradientView struct {
	Block        int     `json:"block"`
	ParameterRMS float64 `json:"parameterRms"`
	InputRMS     float64 `json:"inputRms"`
}
type Summary struct {
	ParameterCount     int           `json:"parameterCount"`
	BestEpoch          int           `json:"bestEpoch"`
	Accepted           bool          `json:"accepted"`
	AcceptanceMSE      float64       `json:"acceptanceMse"`
	InitialValidation  neuro.Metrics `json:"initialValidation"`
	BestTrain          neuro.Metrics `json:"bestTrain"`
	BestValidation     neuro.Metrics `json:"bestValidation"`
	Test               neuro.Metrics `json:"test"`
	IdentityValidation neuro.Metrics `json:"identityValidation"`
	AffineValidation   neuro.Metrics `json:"affineValidation"`
	IdentityTest       neuro.Metrics `json:"identityTest"`
	AffineTest         neuro.Metrics `json:"affineTest"`
	AffineCoefficients [][]float64   `json:"affineCoefficients"`
	TrainingSeconds    float64       `json:"trainingSeconds"`
	OptimizerSteps     int64         `json:"optimizerSteps"`
	DiagnosticsSamples int           `json:"diagnosticsSamples"`
	CheckpointNote     string        `json:"checkpointNote"`
}
type ReportData struct {
	Config         Config         `json:"config"`
	Summary        Summary        `json:"summary"`
	History        []EpochRecord  `json:"history"`
	Grid           []PointView    `json:"grid"`
	ActivationSide int            `json:"activationSide"`
	Activations    [][]float64    `json:"activations"`
	Gradients      []GradientView `json:"gradients"`
	Trajectories   [][][]float64  `json:"trajectories"`
}

const help = `Neuro — CPU neural network written from scratch (Go standard library).
train [--out runs/name] [--epochs 200] [--width 20] [--blocks 10]
      [--activation Swish|Relu|Tanh|LeakyRelu] [--optimizer Adam|Sgd]
      [--alpha 0.1] [--lr 0.001] [--batch 64] [--train 8192]
      [--validation 2048] [--test 2048] [--steps 20] [--h 0.2] [--grid 101]
      [--weight-seed 42] [--shuffle-seed 4004] [--train-seed 1001]
      [--validation-seed 2002] [--test-seed 3003]
evaluate <run-directory>   Reload best weights; reproduce held-out metrics.
report <run-directory>     Rebuild local HTML from saved report-data.json.
Output directory must not exist. No browser or server is launched.
`

func Run(args []string, out io.Writer) error {
	if len(args) == 0 || args[0] == "--help" || args[0] == "help" {
		_, err := fmt.Fprint(out, help)
		return err
	}
	switch args[0] {
	case "train":
		c, err := ParseConfig(args[1:])
		if err != nil {
			return err
		}
		return Train(c, out)
	case "evaluate":
		if len(args) != 2 {
			return fmt.Errorf("evaluate needs one run directory")
		}
		return EvaluateSaved(args[1], out)
	case "report":
		if len(args) != 2 {
			return fmt.Errorf("report needs one run directory")
		}
		return RegenerateReport(args[1], out)
	default:
		return fmt.Errorf("use train, evaluate or report; see --help")
	}
}

func saveJSON(path string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func readJSON(path string, value any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}

func Train(c Config, out io.Writer) error {
	if err := c.Validate(); err != nil {
		return err
	}
	// Mkdir is exclusive. MkdirAll alone would silently reuse existing results.
	if err := os.MkdirAll(filepath.Dir(c.Output), 0755); err != nil {
		return err
	}
	if err := os.Mkdir(c.Output, 0755); err != nil {
		return fmt.Errorf("create new output directory: %w", err)
	}
	save := func(name string, v any) error { return saveJSON(filepath.Join(c.Output, name), v) }
	if err := save("config.json", c); err != nil {
		return err
	}
	if err := save("environment.json", map[string]any{
		"runtime": runtime.Version(), "os": runtime.GOOS, "architecture": runtime.GOARCH, "cpuThreads": runtime.NumCPU(),
		"git": gitState(), "generatedUtc": time.Now().UTC(), "motionVersion": neuro.MotionVersion,
		"randomVersion": "splitmix64-v1", "normalization": "divide coordinates by pi",
		"initialization": "uniform +/- sqrt(6/(fanIn+fanOut)), zero bias", "singleThread": true,
	}); err != nil {
		return err
	}
	train := neuro.Generate(c.TrainCount, c.TrainSeed, c.Steps, c.H)
	validation := neuro.Generate(c.ValidationCount, c.ValidationSeed, c.Steps, c.H)
	affine, err := neuro.FitAffine(train)
	if err != nil {
		return err
	}
	identity := func(x []float64) []float64 { return x }
	identityVal, err := neuro.Measure(validation, identity)
	if err != nil {
		return err
	}
	affineVal, err := neuro.Measure(validation, affine.Predict)
	if err != nil {
		return err
	}
	n := neuro.NewNetwork(c.Width, c.Blocks, c.Activation, c.Alpha, c.WeightSeed)
	o := neuro.NewOptimizer(n.Parameters, c.Optimizer, c.LearningRate)
	shuffle := neuro.RandomStream{State: c.ShuffleSeed}
	order := make([]int, len(train))
	for i := range order {
		order[i] = i
	}
	initialVal, err := neuro.Measure(validation, n.Forward)
	if err != nil {
		return err
	}
	initialTrain, err := neuro.Measure(train, n.Forward)
	if err != nil {
		return err
	}
	history := []EpochRecord{{Epoch: 0, Train: initialTrain, Validation: initialVal}}
	bestMSE, bestEpoch, best := initialVal.MSE, 0, n.Snapshot()
	csvFile, err := os.Create(filepath.Join(c.Output, "metrics.csv"))
	if err != nil {
		return err
	}
	defer csvFile.Close()
	w := csv.NewWriter(csvFile)
	if err := w.Write([]string{"epoch", "train_mse", "validation_mse", "train_distance", "validation_distance", "train_seconds", "evaluation_seconds"}); err != nil {
		return err
	}
	if err := writeRow(w, history[0]); err != nil {
		return err
	}
	fmt.Fprintf(out, "Parameters=%d; width=%d; blocks=%d; activation=%s; optimizer=%s\n", n.ParameterCount(), c.Width, c.Blocks, c.Activation, c.Optimizer)
	fmt.Fprintf(out, "Validation baselines: identity=%.6g, affine=%.6g; initial=%.6g; acceptance<=%g\n", identityVal.MSE, affineVal.MSE, initialVal.MSE, c.AcceptanceMSE)
	started := time.Now()
	for epoch := 1; epoch <= c.Epochs; epoch++ {
		shuffle.Shuffle(order)
		timer := time.Now()
		for start := 0; start < len(order); start += c.BatchSize {
			count := min(c.BatchSize, len(order)-start)
			loss := neuro.Accumulate(n, train, order, start, count)
			if !neuro.Finite(loss) {
				return fmt.Errorf("non-finite batch loss at epoch %d", epoch)
			}
			if err := o.Step(); err != nil {
				return fmt.Errorf("epoch %d: %w", epoch, err)
			}
		}
		trainSeconds := time.Since(timer).Seconds()
		timer = time.Now()
		trainMetric, err := neuro.Measure(train, n.Forward)
		if err != nil {
			return err
		}
		valMetric, err := neuro.Measure(validation, n.Forward)
		if err != nil {
			return err
		}
		r := EpochRecord{epoch, trainMetric, valMetric, trainSeconds, time.Since(timer).Seconds()}
		history = append(history, r)
		if err := writeRow(w, r); err != nil {
			return err
		}
		if valMetric.MSE < bestMSE {
			bestMSE, bestEpoch, best = valMetric.MSE, epoch, n.Snapshot()
		}
		if epoch <= 3 || epoch%10 == 0 || epoch == c.Epochs {
			fmt.Fprintf(out, "epoch=%d train=%.6g val=%.6g distance=%.4f train_s=%.3f\n", epoch, trainMetric.MSE, valMetric.MSE, valMetric.MeanDistance, trainSeconds)
		}
	}
	seconds := time.Since(started).Seconds()
	if err := save("last-weights.json", WeightFile{1, c, c.Epochs, n.Snapshot()}); err != nil {
		return err
	}
	if err := save("best-weights.json", WeightFile{1, c, bestEpoch, best}); err != nil {
		return err
	}
	if err := n.Restore(best); err != nil {
		return err
	}
	bestTrain, err := neuro.Measure(train, n.Forward)
	if err != nil {
		return err
	}
	bestVal, err := neuro.Measure(validation, n.Forward)
	if err != nil {
		return err
	}
	// Test is created only after selecting weights on validation.
	test := neuro.Generate(c.TestCount, c.TestSeed, c.Steps, c.H)
	testMetric, err := neuro.Measure(test, n.Forward)
	if err != nil {
		return err
	}
	identityTest, err := neuro.Measure(test, identity)
	if err != nil {
		return err
	}
	affineTest, err := neuro.Measure(test, affine.Predict)
	if err != nil {
		return err
	}
	accepted := bestVal.MSE <= c.AcceptanceMSE && bestVal.MSE < initialVal.MSE && bestVal.MSE < identityVal.MSE && bestVal.MSE < affineVal.MSE
	summary := Summary{n.ParameterCount(), bestEpoch, accepted, c.AcceptanceMSE, initialVal, bestTrain, bestVal, testMetric,
		identityVal, affineVal, identityTest, affineTest, affine.Coefficients, seconds, o.StepCount, min(64, len(validation)),
		"Weights for evaluation only; optimizer/RNG state not saved, resume unsupported."}
	report := ReportData{Config: c, Summary: summary, History: history, Gradients: diagnostics(n, validation), ActivationSide: 21}
	grid := neuro.Grid(c.GridSide, c.Steps, c.H)
	report.Grid = make([]PointView, len(grid))
	for i, s := range grid {
		y := n.Forward(s.Input)
		report.Grid[i] = PointView{s.Input[0] * math.Pi, s.Input[1] * math.Pi, s.Target[0] * math.Pi, s.Target[1] * math.Pi, y[0] * math.Pi, y[1] * math.Pi, math.Sqrt(2*neuro.MSE(y, s.Target)) * math.Pi}
	}
	activationGrid := neuro.Grid(report.ActivationSide, c.Steps, c.H)
	report.Activations = make([][]float64, c.Blocks)
	for b := range report.Activations {
		report.Activations[b] = make([]float64, len(activationGrid)*c.Width)
	}
	for i, s := range activationGrid {
		n.Forward(s.Input)
		for b, block := range n.Blocks {
			copy(report.Activations[b][i*c.Width:], block.Activated)
		}
	}
	for _, initial := range [][2]float64{{0.4, 0.8}, {-1.8, 1}, {2, -0.6}, {-2.4, -2}} {
		x, y := initial[0], initial[1]
		path := [][]float64{{x, y}}
		for s := 0; s < c.Steps; s++ {
			x, y = neuro.MotionStep(x, y, c.H)
			path = append(path, []float64{x, y})
		}
		report.Trajectories = append(report.Trajectories, path)
	}
	for name, value := range map[string]any{"summary.json": summary, "history.json": history, "report-data.json": report} {
		if err := save(name, value); err != nil {
			return err
		}
	}
	if err := WriteReport(filepath.Join(c.Output, "report.html"), report); err != nil {
		return err
	}
	fmt.Fprintf(out, "Best epoch=%d; validation=%.6g; test=%.6g; accepted=%t; seconds=%.2f\n", bestEpoch, bestVal.MSE, testMetric.MSE, accepted, seconds)
	path, _ := filepath.Abs(filepath.Join(c.Output, "report.html"))
	fmt.Fprintln(out, "Report:", path)
	return nil
}

func EvaluateSaved(directory string, out io.Writer) error {
	var file WeightFile
	if err := readJSON(filepath.Join(directory, "best-weights.json"), &file); err != nil {
		return err
	}
	if file.FormatVersion != 1 {
		return fmt.Errorf("unsupported checkpoint version")
	}
	c := file.Configuration
	if err := c.Validate(); err != nil {
		return err
	}
	if file.Epoch < 0 || file.Epoch > c.Epochs {
		return fmt.Errorf("invalid checkpoint epoch")
	}
	n := neuro.NewNetwork(c.Width, c.Blocks, c.Activation, c.Alpha, c.WeightSeed)
	if err := n.Restore(file.Parameters); err != nil {
		return err
	}
	validation, err := neuro.Measure(neuro.Generate(c.ValidationCount, c.ValidationSeed, c.Steps, c.H), n.Forward)
	if err != nil {
		return err
	}
	test, err := neuro.Measure(neuro.Generate(c.TestCount, c.TestSeed, c.Steps, c.H), n.Forward)
	if err != nil {
		return err
	}
	return json.NewEncoder(out).Encode(map[string]any{"epoch": file.Epoch, "validation": validation, "test": test})
}

func diagnostics(n *neuro.Network, data []neuro.Sample) []GradientView {
	count := min(64, len(data))
	inputSquares := make([]float64, len(n.Blocks))
	n.ZeroGradients()
	for i := 0; i < count; i++ {
		s := data[i]
		y := n.Forward(s.Input)
		n.Backward(neuro.LossGradient(y, s.Target, count))
		for b, block := range n.Blocks {
			g := block.LastInputGradientRMS * float64(count)
			inputSquares[b] += g * g
		}
	}
	result := make([]GradientView, len(n.Blocks))
	for b, block := range n.Blocks {
		countP := 0
		sum := 0.0
		for _, p := range []*neuro.Parameter{block.First.Weights, block.First.Bias, block.Second.Weights, block.Second.Bias} {
			local := 0.0
			for _, g := range p.Gradients {
				local += g * g
			}
			sum += local
			countP += len(p.Gradients)
		}
		result[b] = GradientView{b, math.Sqrt(sum / float64(countP)), math.Sqrt(inputSquares[b] / float64(count))}
	}
	return result
}

func writeRow(w *csv.Writer, r EpochRecord) error {
	num := func(x float64) string { return strconv.FormatFloat(x, 'g', -1, 64) }
	if err := w.Write([]string{strconv.Itoa(r.Epoch), num(r.Train.MSE), num(r.Validation.MSE), num(r.Train.MeanDistance), num(r.Validation.MeanDistance), num(r.TrainSeconds), num(r.EvaluationSeconds)}); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}

func gitState() any {
	git := func(args ...string) (string, error) {
		data, err := exec.Command("git", args...).Output()
		return strings.TrimSpace(string(data)), err
	}
	revision, err := git("rev-parse", "HEAD")
	if err != nil {
		return map[string]any{"revision": "unavailable", "dirty": true}
	}
	status, err := git("status", "--porcelain")
	return map[string]any{"revision": revision, "dirty": err != nil || status != ""}
}
