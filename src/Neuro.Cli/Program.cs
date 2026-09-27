using System.Globalization;
using Neuro.Cli;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
if (args.Length == 0 || args[0] is "--help" or "help")
{
    Console.WriteLine("""
        Neuro — CPU neural network written from scratch (.NET standard library).
        train [--out runs/name] [--epochs 200] [--width 20] [--blocks 10]
              [--activation Swish|Relu|Tanh|LeakyRelu] [--optimizer Adam|Sgd]
              [--alpha 0.1] [--lr 0.001] [--batch 64] [--train 8192]
              [--validation 2048] [--test 2048] [--steps 20] [--h 0.2] [--grid 101]
              [--weight-seed 42] [--shuffle-seed 4004] [--train-seed 1001]
              [--validation-seed 2002] [--test-seed 3003]
        evaluate <run-directory>   Reload best weights; reproduce held-out metrics.
        report <run-directory>     Rebuild local HTML from saved report-data.json.
        Output directory must not exist. No browser or server is launched.
        """);
    return 0;
}
try
{
    if (args[0] == "train") { Training.Run(Configuration.Parse(args[1..])); return 0; }
    if (args[0] == "evaluate" && args.Length == 2) { Training.EvaluateSaved(args[1]); return 0; }
    if (args[0] == "report" && args.Length == 2) { Report.Regenerate(args[1]); return 0; }
    throw new ArgumentException("Use train, evaluate or report; see --help.");
}
catch (Exception e)
{
    Console.Error.WriteLine($"ERROR: {e.Message}");
    return 1;
}
