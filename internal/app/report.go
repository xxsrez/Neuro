package app

import (
	"encoding/json"
	"fmt"
	"html"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"neuro/assets"
)

func number(x float64) string { return fmt.Sprintf("%.6g", x) }

func RegenerateReport(directory string, out io.Writer) error {
	var data ReportData
	if err := readJSON(filepath.Join(directory, "report-data.json"), &data); err != nil {
		return err
	}
	if err := WriteReport(filepath.Join(directory, "report.html"), data); err != nil {
		return err
	}
	path, _ := filepath.Abs(filepath.Join(directory, "report.html"))
	_, err := fmt.Fprintln(out, path)
	return err
}

func WriteReport(path string, d ReportData) error {
	if err := d.Config.Validate(); err != nil {
		return err
	}
	c, s := d.Config, d.Summary
	if len(d.History) == 0 || len(d.Grid) != c.GridSide*c.GridSide || len(d.Activations) != c.Blocks || d.ActivationSide < 2 || len(d.Gradients) != c.Blocks {
		return fmt.Errorf("invalid report dimensions")
	}
	for _, values := range d.Activations {
		if len(values) != d.ActivationSide*d.ActivationSide*c.Width {
			return fmt.Errorf("invalid activation dimensions")
		}
	}
	// Marshal first to reject non-finite numbers before constructing any HTML.
	embedded, err := json.Marshal(d)
	if err != nil {
		return err
	}
	configJSON, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	var content strings.Builder
	fmt.Fprintf(&content, `<p class="eyebrow">NEURO · ЛОКАЛЬНЫЙ ОПЫТ</p><h1>Как сеть учится движению точки</h1><p>Ширина %d · блоков %d · %s · %s · параметров: %d.</p>`, c.Width, c.Blocks, html.EscapeString(string(c.Activation)), html.EscapeString(string(c.Optimizer)), s.ParameterCount)
	accepted := "не достигнут"
	if s.Accepted {
		accepted = "достигнут"
	}
	fmt.Fprintf(&content, `<p>Лучшая эпоха: <strong>%d</strong>. MSE на валидации: <strong>%s</strong>. Тестовая MSE: <strong>%s</strong>. Критерий первого опыта: <strong>%s</strong>.</p>`, s.BestEpoch, number(s.BestValidation.MSE), number(s.Test.MSE), accepted)
	content.WriteString(`<p class="muted">Лучшие веса выбраны по валидации, тест оценён после выбора. Один запуск не доказывает преимущество глубины или активации. MSE считается после деления координат на π; расстояние — в исходных координатах.</p><div class="table-wrap"><table><thead><tr><th>Модель</th><th>MSE, валидация</th><th>Среднее расстояние</th></tr></thead><tbody>`)
	for _, row := range []struct {
		title         string
		mse, distance float64
	}{
		{"Начальные веса", s.InitialValidation.MSE, s.InitialValidation.MeanDistance}, {"Без движения", s.IdentityValidation.MSE, s.IdentityValidation.MeanDistance},
		{"Аффинная модель", s.AffineValidation.MSE, s.AffineValidation.MeanDistance}, {"Лучшие веса сети", s.BestValidation.MSE, s.BestValidation.MeanDistance},
	} {
		fmt.Fprintf(&content, "<tr><td>%s</td><td>%s</td><td>%s</td></tr>", row.title, number(row.mse), number(row.distance))
	}
	content.WriteString(`</tbody></table></div><section><h2>Ошибка по эпохам</h2><p>Обучающая — синяя, валидационная — оранжевая. Обе рассчитаны после эпохи при одинаковых весах. Ниже — линейная шкала.</p>`)
	content.WriteString(lossSVG(d.History))
	content.WriteString(`</section><section><h2>Где сеть ошибается</h2><p>Карта для лучших весов. По осям — начальные координаты от −π до π. Синий — малая ошибка, красный — большая. Максимальное расстояние на сетке: `)
	maxError := 0.0
	for _, p := range d.Grid {
		maxError = math.Max(maxError, p.Error)
	}
	content.WriteString(number(maxError) + ".</p>" + heatSVG(d.Grid, c.GridSide) + "</section>")
	fmt.Fprintf(&content, `<section><h2>Градиент по глубине</h2><p>RMS градиента параметров каждого блока на первых %d точках валидации. Измерение на лучших весах до обновления, с усреднением по диагностическому пакету.</p>%s</section>`, s.DiagnosticsSamples, gradientSVG(d.Gradients))
	content.WriteString(`<details><summary>Конфигурация опыта</summary><pre>` + html.EscapeString(string(configJSON)) + `</pre></details>`)
	if c.Blocks > 0 {
		values := d.Activations[0]
		scale := maxAbsolute(values)
		fmt.Fprintf(&content, `<section><h2>Активности первого блока</h2><p>Все %d нейронов, сетка %d × %d; общий диапазон цвета ±%s. Для других блоков используйте интерактивный выбор ниже.</p><div class="tiles">`, c.Width, d.ActivationSide, d.ActivationSide, number(scale))
		for neuron := 0; neuron < c.Width; neuron++ {
			fmt.Fprintf(&content, `<div class="tile"><p>Нейрон %d</p>%s</div>`, neuron+1, activationSVG(values, d.ActivationSide, c.Width, neuron, scale))
		}
		content.WriteString("</div></section>")
	}
	// encoding/json escapes '<', including any '</script>' in user-controlled strings.
	page := strings.ReplaceAll(assets.ReportTemplate, "@@STATIC@@", content.String())
	page = strings.ReplaceAll(page, "@@DATA@@", string(embedded))
	return os.WriteFile(path, []byte(page), 0644)
}

func frame(body, caption string) string {
	return `<svg viewBox="0 0 720 330" role="img" aria-label="` + html.EscapeString(caption) + `"><rect x="55" y="15" width="640" height="275" fill="none" stroke="#94a3b8"/>` + body + `</svg>`
}
func lossSVG(history []EpochRecord) string {
	maximum := 1e-12
	for _, r := range history {
		maximum = math.Max(maximum, math.Max(r.Train.MSE, r.Validation.MSE))
	}
	var body strings.Builder
	fmt.Fprintf(&body, `<text x="5" y="25">%s</text><text x="30" y="290">0</text><text x="55" y="315">0</text><text x="600" y="315">Эпоха %d</text>`, number(maximum), history[len(history)-1].Epoch)
	for line, color := range []string{"#2563eb", "#ea580c"} {
		fmt.Fprintf(&body, `<polyline fill="none" stroke="%s" stroke-width="2" points="`, color)
		for _, r := range history {
			value := r.Train.MSE
			if line == 1 {
				value = r.Validation.MSE
			}
			fmt.Fprintf(&body, "%s,%s ", number(55+640*float64(r.Epoch)/float64(max(1, history[len(history)-1].Epoch))), number(290-275*value/maximum))
		}
		body.WriteString(`"/>`)
	}
	return frame(body.String(), "MSE обучения и валидации по эпохам")
}
func heatSVG(grid []PointView, side int) string {
	maximum := 1e-12
	for _, p := range grid {
		maximum = math.Max(maximum, p.Error)
	}
	var body strings.Builder
	body.WriteString(`<svg viewBox="0 0 360 330" style="max-width:480px" role="img" aria-label="Карта расстояний до правильной конечной точки">`)
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			t := max(0.0, min(1.0, grid[y*side+x].Error/maximum))
			fmt.Fprintf(&body, `<rect x="%s" y="%s" width="%s" height="%s" fill="rgb(%d,%d,%d)"/>`, number(55+275*float64(x)/float64(side)), number(15+275*float64(side-1-y)/float64(side)), number(275/float64(side)+0.1), number(275/float64(side)+0.1), int(40+210*t), int(120-60*t), int(220-180*t))
		}
	}
	body.WriteString(`<text x="55" y="315">−π</text><text x="310" y="315">π · x₀</text><text x="8" y="25">π</text><text x="5" y="285">−π</text></svg>`)
	return body.String()
}
func maxAbsolute(values []float64) float64 {
	scale := 1e-12
	for _, x := range values {
		scale = math.Max(scale, math.Abs(x))
	}
	return scale
}
func activationSVG(values []float64, side, width, neuron int, scale float64) string {
	var body strings.Builder
	fmt.Fprintf(&body, `<svg viewBox="0 0 %d %d" role="img" aria-label="Активность нейрона %d первого блока">`, side, side, neuron+1)
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			v := values[(y*side+x)*width+neuron]
			t := math.Min(1, math.Abs(v)/scale)
			r, g, b := 245, int(240-180*t), int(240-190*t)
			if v < 0 {
				r, g, b = int(240-200*t), int(240-120*t), int(245-20*t)
			}
			fmt.Fprintf(&body, `<rect x="%d" y="%d" width="1.05" height="1.05" fill="rgb(%d,%d,%d)"/>`, x, side-1-y, r, g, b)
		}
	}
	body.WriteString("</svg>")
	return body.String()
}
func gradientSVG(gradients []GradientView) string {
	maximum := 1e-12
	for _, g := range gradients {
		maximum = math.Max(maximum, g.ParameterRMS)
	}
	var body strings.Builder
	fmt.Fprintf(&body, `<text x="5" y="25">%s</text>`, number(maximum))
	for i, g := range gradients {
		x := 60 + 630*float64(i)/float64(len(gradients))
		h := 270 * g.ParameterRMS / maximum
		fmt.Fprintf(&body, `<rect x="%s" y="%s" width="%s" height="%s" fill="#2563eb"/><text x="%s" y="315">%d</text>`, number(x), number(290-h), number(500/float64(len(gradients))), number(h), number(x), i+1)
	}
	return frame(body.String(), "Градиенты параметров по блокам")
}
