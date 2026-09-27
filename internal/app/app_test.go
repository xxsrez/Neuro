package app

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
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
