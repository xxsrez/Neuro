package app

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"neuro/internal/neuro"
)

func TestConfig(t *testing.T) {
	c, err := ParseConfig([]string{"--activation", "sWiSh", "--optimizer", "sgd", "--epochs", "2"})
	if err != nil || c.Activation != neuro.Swish || c.Optimizer != neuro.SGD || c.Epochs != 2 {
		t.Fatal(c, err)
	}
	for _, args := range [][]string{{"--batch", "0"}, {"--unknown", "1"}, {"--lr", "NaN"}, {"--alpha", "Inf"}, {"--train-seed", "2002"}, {"--activation", "fake"}, {"--epochs"}, {"extra"}, {"--out", ""}} {
		if _, err := ParseConfig(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestCLIEndToEnd(t *testing.T) {
	c := DefaultConfig()
	c.Output = filepath.Join(t.TempDir(), "run")
	c.Width = 4
	c.Blocks = 2
	c.Epochs = 2
	c.TrainCount = 65
	c.ValidationCount = 16
	c.TestCount = 16
	c.GridSide = 5
	var out bytes.Buffer
	if err := Train(c, &out); err != nil {
		t.Fatal(err)
	}
	var summary Summary
	if err := readJSON(filepath.Join(c.Output, "summary.json"), &summary); err != nil {
		t.Fatal(err)
	}
	if summary.OptimizerSteps != 4 {
		t.Fatal("incomplete final batch was lost")
	}
	var eval bytes.Buffer
	if err := EvaluateSaved(c.Output, &eval); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Epoch            int `json:"epoch"`
		Validation, Test neuro.Metrics
	}
	if err := json.Unmarshal(eval.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Validation != summary.BestValidation || result.Test != summary.Test || result.Epoch != summary.BestEpoch {
		t.Fatal("loaded metrics differ")
	}
	if err := Train(c, &out); err == nil {
		t.Fatal("overwrote an existing run")
	}
	var report ReportData
	if err := readJSON(filepath.Join(c.Output, "report-data.json"), &report); err != nil {
		t.Fatal(err)
	}
	page, err := os.ReadFile(filepath.Join(c.Output, "report.html"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(page, []byte("@@DATA@@")) || !bytes.Contains(page, []byte("<svg ")) {
		t.Fatal("bad report")
	}
	if err := RegenerateReport(c.Output, &out); err != nil {
		t.Fatal(err)
	}
	page2, _ := os.ReadFile(filepath.Join(c.Output, "report.html"))
	if !bytes.Equal(page, page2) {
		t.Fatal("report regeneration differs")
	}
	// User strings cannot close the JSON script or HTML pre element.
	report.Config.Output = "</script><script>alert(1)</script>"
	safe := filepath.Join(t.TempDir(), "safe.html")
	if err := WriteReport(safe, report); err != nil {
		t.Fatal(err)
	}
	page, _ = os.ReadFile(safe)
	if strings.Contains(string(page), report.Config.Output) {
		t.Fatal("unsafe HTML insertion")
	}
}

func TestZeroBlocks(t *testing.T) {
	c := DefaultConfig()
	c.Output = filepath.Join(t.TempDir(), "zero")
	c.Blocks = 0
	c.Epochs = 1
	c.TrainCount = 8
	c.ValidationCount = 4
	c.TestCount = 4
	c.GridSide = 2
	if err := Train(c, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	var report ReportData
	if err := readJSON(filepath.Join(c.Output, "report-data.json"), &report); err != nil {
		t.Fatal(err)
	}
	if report.Activations == nil || report.Gradients == nil || len(report.Activations) != 0 || len(report.Gradients) != 0 {
		t.Fatal("empty arrays must serialize as []")
	}
}

// Optional local parity test uses original C# artifacts if available. A fresh
// clone can run all other tests without generated data or .NET installed.
func TestOriginalCheckpointParity(t *testing.T) {
	directory := filepath.Join("..", "..", "runs", "prototype")
	if _, err := os.Stat(filepath.Join(directory, "best-weights.json")); err != nil {
		t.Skip("original C# run not present")
	}
	var saved WeightFile
	if err := readJSON(filepath.Join(directory, "best-weights.json"), &saved); err != nil {
		t.Fatal(err)
	}
	var report ReportData
	if err := readJSON(filepath.Join(directory, "report-data.json"), &report); err != nil {
		t.Fatal(err)
	}
	c := saved.Configuration
	n := neuro.NewNetwork(c.Width, c.Blocks, c.Activation, c.Alpha, c.WeightSeed)
	if err := n.Restore(saved.Parameters); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(report.Grid); i += 97 {
		p := report.Grid[i]
		y := n.Forward([]float64{p.X / math.Pi, p.Y / math.Pi})
		if math.Abs(y[0]*math.Pi-p.PredictedX) > 1e-11 || math.Abs(y[1]*math.Pi-p.PredictedY) > 1e-11 {
			t.Fatalf("cross-language prediction mismatch at %d", i)
		}
	}
	var out bytes.Buffer
	if err := EvaluateSaved(directory, &out); err != nil {
		t.Fatal(err)
	}
	var eval struct{ Validation, Test neuro.Metrics }
	if err := json.Unmarshal(out.Bytes(), &eval); err != nil {
		t.Fatal(err)
	}
	if math.Abs(eval.Validation.MSE-report.Summary.BestValidation.MSE) > 1e-12 || math.Abs(eval.Test.MSE-report.Summary.Test.MSE) > 1e-12 {
		t.Fatal("cross-language metrics differ")
	}
	before := n.Snapshot()
	if err := n.Restore(saved.Parameters); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, n.Snapshot()) {
		t.Fatal("unexpected checkpoint mutation")
	}
}
