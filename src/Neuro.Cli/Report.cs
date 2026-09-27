using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;

namespace Neuro.Cli;

public static class Report
{
    public static void Regenerate(string directory)
    {
        string data = File.ReadAllText(Path.Combine(directory, "report-data.json"));
        var root = JsonSerializer.Deserialize<JsonElement>(data);
        var c = root.GetProperty("config").Deserialize<Configuration>(Training.Json) ?? throw new InvalidDataException("Invalid config");
        var history = root.GetProperty("history").Deserialize<List<EpochRecord>>(Training.Json) ?? throw new InvalidDataException("Invalid history");
        var grid = root.GetProperty("grid").Deserialize<PointView[]>(Training.Json) ?? throw new InvalidDataException("Invalid grid");
        var gradients = root.GetProperty("gradients").Deserialize<GradientView[]>(Training.Json) ?? throw new InvalidDataException("Invalid diagnostics");
        Write(Path.Combine(directory, "report.html"), c, history, grid, gradients, root.GetProperty("summary"), data);
        Console.WriteLine(Path.GetFullPath(Path.Combine(directory, "report.html")));
    }
    private static string N(double value) => value.ToString("G6", CultureInfo.InvariantCulture);
    private static string E(string value) => WebUtility.HtmlEncode(value);
    public static void Write(string path, Configuration c, List<EpochRecord> history, PointView[] grid,
        GradientView[] gradients, object summary, string data)
    {
        var s = JsonSerializer.SerializeToElement(summary, Training.Json);
        string table = "<table><thead><tr><th>Модель</th><th>MSE, валидация</th><th>Среднее расстояние</th></tr></thead><tbody>";
        foreach (var (key, title) in new[] { ("initialValidation", "Начальные веса"), ("identityValidation", "Без движения"), ("affineValidation", "Аффинная модель"), ("bestValidation", "Лучшие веса сети") })
            table += $"<tr><td>{title}</td><td>{N(s.GetProperty(key).GetProperty("mse").GetDouble())}</td><td>{N(s.GetProperty(key).GetProperty("meanDistance").GetDouble())}</td></tr>";
        table += "</tbody></table>";
        var content = $"""
            <p class="eyebrow">NEURO · ЛОКАЛЬНЫЙ ОПЫТ</p>
            <h1>Как сеть учится движению точки</h1>
            <p>Ширина {c.Width} · блоков {c.Blocks} · {E(c.Activation.ToString())} · {E(c.Optimizer.ToString())} · {s.GetProperty("parameterCount")} параметра.</p>
            <p>Лучшая эпоха: <strong>{s.GetProperty("bestEpoch")}</strong>. MSE на валидации:
            <strong>{N(s.GetProperty("bestValidation").GetProperty("mse").GetDouble())}</strong>.
            Тестовая MSE: <strong>{N(s.GetProperty("test").GetProperty("mse").GetDouble())}</strong>.
            Критерий первого опыта: <strong>{(s.GetProperty("accepted").GetBoolean() ? "достигнут" : "не достигнут")}</strong>.</p>
            <p class="muted">Лучшие веса выбраны по валидации, тест оценён после выбора. Один запуск не доказывает преимущество глубины или активации. MSE считается после деления координат на π; расстояние — в исходных координатах.</p>
            <div class="table-wrap">{table}</div>
            <section><h2>Ошибка по эпохам</h2><p>Обучающая — синяя, валидационная — оранжевая. Обе рассчитаны после эпохи при одинаковых весах. Ниже — линейная шкала.</p>{LossSvg(history)}</section>
            <section><h2>Где сеть ошибается</h2><p>Карта для лучших весов. По осям — начальные координаты от −π до π. Синий — малая ошибка, красный — большая. Максимальное расстояние на сетке: {N(grid.Max(p => p.Error))}.</p>{HeatSvg(grid, c.GridSide)}</section>
            <section><h2>Градиент по глубине</h2><p>Среднеквадратичная величина градиента параметров каждого блока на первых {s.GetProperty("diagnosticsSamples")} точках валидации. Измерение на лучших весах, до обновления, с усреднением по диагностическому пакету.</p>{GradientSvg(gradients)}</section>
            <details><summary>Конфигурация опыта</summary><pre>{E(JsonSerializer.Serialize(c, new JsonSerializerOptions(Training.Json) { WriteIndented = true }))}</pre></details>
            """;
        var reportData = JsonSerializer.Deserialize<JsonElement>(data);
        if (c.Blocks > 0)
        {
            var values = reportData.GetProperty("activations")[0].EnumerateArray().Select(v => v.GetDouble()).ToArray();
            int side = reportData.GetProperty("activationSide").GetInt32();
            content += $"<section><h2>Активности первого блока</h2><p>Все {c.Width} нейронов, сетка {side} × {side}; общий диапазон цвета ±{N(values.Max(Math.Abs))}. Для других блоков используйте интерактивный выбор ниже.</p><div class=\"tiles\">";
            for (int neuron = 0; neuron < c.Width; neuron++)
                content += $"<div class=\"tile\"><p>Нейрон {neuron + 1}</p>{ActivationSvg(values, side, c.Width, neuron)}</div>";
            content += "</div></section>";
        }
        string template = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "report.html"));
        // System.Text.Json's default encoder escapes '<', so data cannot close the script element.
        File.WriteAllText(path, template.Replace("@@STATIC@@", content, StringComparison.Ordinal)
            .Replace("@@DATA@@", data, StringComparison.Ordinal));
    }
    private static string Frame(string body, string caption) =>
        $"<svg viewBox=\"0 0 720 330\" role=\"img\" aria-label=\"{E(caption)}\"><rect x=\"55\" y=\"15\" width=\"640\" height=\"275\" fill=\"none\" stroke=\"#94a3b8\"/>{body}</svg>";
    private static string LossSvg(List<EpochRecord> history)
    {
        double max = Math.Max(1e-12, history.Max(r => Math.Max(r.Train.Mse, r.Validation.Mse)));
        string Line(Func<EpochRecord, double> value, string color)
        {
            string points = string.Join(" ", history.Select(r => $"{N(55 + 640.0 * r.Epoch / Math.Max(1, history[^1].Epoch))},{N(290 - 275 * value(r) / max)}"));
            return $"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\"{points}\"/>";
        }
        return Frame($"<text x=\"5\" y=\"25\">{N(max)}</text><text x=\"30\" y=\"290\">0</text><text x=\"55\" y=\"315\">0</text><text x=\"600\" y=\"315\">Эпоха {history[^1].Epoch}</text>" +
            Line(r => r.Train.Mse, "#2563eb") + Line(r => r.Validation.Mse, "#ea580c"), "MSE обучения и валидации по эпохам");
    }
    private static string HeatSvg(PointView[] grid, int side)
    {
        double max = Math.Max(1e-12, grid.Max(p => p.Error)); var body = new StringBuilder();
        for (int y = 0; y < side; y++) for (int x = 0; x < side; x++)
        {
            double t = Math.Clamp(grid[y * side + x].Error / max, 0, 1);
            string color = $"rgb({(int)(40 + 210 * t)},{(int)(120 - 60 * t)},{(int)(220 - 180 * t)})";
            body.Append($"<rect x=\"{N(55 + 275.0 * x / side)}\" y=\"{N(15 + 275.0 * (side - 1 - y) / side)}\" width=\"{N(275.0 / side + 0.1)}\" height=\"{N(275.0 / side + 0.1)}\" fill=\"{color}\"/>");
        }
        body.Append("<text x=\"55\" y=\"315\">−π</text><text x=\"310\" y=\"315\">π · x₀</text><text x=\"8\" y=\"25\">π</text><text x=\"5\" y=\"285\">−π</text>");
        return $"<svg viewBox=\"0 0 360 330\" style=\"max-width:480px\" role=\"img\" aria-label=\"Карта расстояний до правильной конечной точки\">{body}</svg>";
    }
    private static string ActivationSvg(double[] values, int side, int width, int neuron)
    {
        double scale = Math.Max(1e-12, values.Max(Math.Abs)); var body = new StringBuilder();
        for (int y = 0; y < side; y++) for (int x = 0; x < side; x++)
        {
            double value = values[(y * side + x) * width + neuron], t = Math.Min(1, Math.Abs(value) / scale);
            string color = value < 0 ? $"rgb({(int)(240 - 200 * t)},{(int)(240 - 120 * t)},{(int)(245 - 20 * t)})" : $"rgb(245,{(int)(240 - 180 * t)},{(int)(240 - 190 * t)})";
            body.Append($"<rect x=\"{x}\" y=\"{side - 1 - y}\" width=\"1.05\" height=\"1.05\" fill=\"{color}\"/>");
        }
        return $"<svg viewBox=\"0 0 {side} {side}\" role=\"img\" aria-label=\"Активность нейрона {neuron + 1} первого блока\">{body}</svg>";
    }
    private static string GradientSvg(GradientView[] gradients)
    {
        double max = Math.Max(1e-12, gradients.Select(g => g.ParameterRms).DefaultIfEmpty(0).Max());
        string bars = string.Join("", gradients.Select((g, i) =>
            $"<rect x=\"{N(60 + 630.0 * i / gradients.Length)}\" y=\"{N(290 - 270 * g.ParameterRms / max)}\" width=\"{N(500.0 / gradients.Length)}\" height=\"{N(270 * g.ParameterRms / max)}\" fill=\"#2563eb\"/><text x=\"{N(60 + 630.0 * i / gradients.Length)}\" y=\"315\">{i + 1}</text>"));
        return Frame($"<text x=\"5\" y=\"25\">{N(max)}</text>" + bars, "Градиенты параметров по блокам");
    }
}
