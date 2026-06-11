using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AmDon.Benchmark;

public class BenchmarkAnalyzer
{
    private readonly TokenAnalyzer _analyzer;
    private readonly NimClient _nim;
    private readonly string _embeddingModel;

    public BenchmarkAnalyzer(TokenAnalyzer analyzer, NimClient nim, string embeddingModel)
    {
        _analyzer = analyzer;
        _nim = nim;
        _embeddingModel = embeddingModel;
    }

    public async Task RunAnalysis(int maxSamples = 50)
    {
        var sep = new string('=', 60);
        Console.WriteLine(sep);
        Console.WriteLine("DELTA ANGLE BENCHMARK ANALYSIS");
        Console.WriteLine($"Model: {_embeddingModel}");
        Console.WriteLine(sep);

        var triggerFile = "benchmark_data/test_trigger.json";
        var benignFile = "benchmark_data/test_benign.json";

        if (!File.Exists(triggerFile) || !File.Exists(benignFile))
        {
            Console.WriteLine("\nBenchmark data not found. Run download_benchmarks.py first.");
            return;
        }

        var triggerSamples = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(triggerFile));
        var benignSamples = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(benignFile));

        Console.WriteLine($"\nTest set: {triggerSamples.Count} trigger, {benignSamples.Count} benign");

        if (triggerSamples.Count > maxSamples)
            triggerSamples = triggerSamples.GetRange(0, maxSamples);
        if (benignSamples.Count > maxSamples)
            benignSamples = benignSamples.GetRange(0, maxSamples);

        Console.WriteLine($"Analyzing {triggerSamples.Count} trigger + {benignSamples.Count} benign samples...\n");

        var allResults = new List<AnalysisResult>();

        for (int i = 0; i < triggerSamples.Count; i++)
        {
            var result = await AnalyzeSample(triggerSamples[i], "trigger", "overall");
            allResults.Add(result);
            if ((i + 1) % 10 == 0)
                Console.WriteLine($"  {i + 1}/{triggerSamples.Count} trigger analyzed");
        }

        for (int i = 0; i < benignSamples.Count; i++)
        {
            var result = await AnalyzeSample(benignSamples[i], "benign", "overall");
            allResults.Add(result);
            if ((i + 1) % 10 == 0)
                Console.WriteLine($"  {i + 1}/{benignSamples.Count} benign analyzed");
        }

        // Save per-sample CSV
        var csvPath = $"benchmark_data/results_{SanitizeModelName(_embeddingModel)}.csv";
        SaveCsv(allResults, csvPath);
        Console.WriteLine($"\nSaved per-sample results to {csvPath}");

        // Print summary
        var triggerResults = allResults.FindAll(r => r.Label == "trigger");
        var benignResults = allResults.FindAll(r => r.Label == "benign");

        Console.WriteLine("\n" + sep);
        Console.WriteLine("RESULTS SUMMARY");
        Console.WriteLine(sep);

        ComputeStatistics(triggerResults, "Trigger (Injection)");
        ComputeStatistics(benignResults, "Benign (Normal)");
        FindOptimalThreshold(triggerResults, benignResults);
    }

    public async Task RunMultiModelAnalysis(List<string> models, int maxSamples = 30)
    {
        var sep = new string('=', 60);
        Console.WriteLine(sep);
        Console.WriteLine("MULTI-MODEL BENCHMARK ANALYSIS");
        Console.WriteLine(sep);

        var triggerFile = "benchmark_data/test_trigger.json";
        var benignFile = "benchmark_data/test_benign.json";

        if (!File.Exists(triggerFile) || !File.Exists(benignFile))
        {
            Console.WriteLine("\nBenchmark data not found. Run download_benchmarks.py first.");
            return;
        }

        var triggerSamples = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(triggerFile));
        var benignSamples = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(benignFile));

        if (triggerSamples.Count > maxSamples)
            triggerSamples = triggerSamples.GetRange(0, maxSamples);
        if (benignSamples.Count > maxSamples)
            benignSamples = benignSamples.GetRange(0, maxSamples);

        Console.WriteLine($"\nSamples per model: {triggerSamples.Count} trigger + {benignSamples.Count} benign");
        Console.WriteLine($"Models to test: {models.Count}\n");

        // Create analyzer for each model
        var allModelResults = new Dictionary<string, List<AnalysisResult>>();

        foreach (var model in models)
        {
            Console.WriteLine($"\n--- Testing {model} ---");
            var analyzer = new TokenAnalyzer(_nim, model);
            var results = new List<AnalysisResult>();

            for (int i = 0; i < triggerSamples.Count; i++)
            {
                var result = await AnalyzeSampleWithModel(analyzer, triggerSamples[i], "trigger", "overall", model);
                results.Add(result);
                if ((i + 1) % 10 == 0)
                    Console.WriteLine($"  {i + 1}/{triggerSamples.Count} trigger");
            }

            for (int i = 0; i < benignSamples.Count; i++)
            {
                var result = await AnalyzeSampleWithModel(analyzer, benignSamples[i], "benign", "overall", model);
                results.Add(result);
                if ((i + 1) % 10 == 0)
                    Console.WriteLine($"  {i + 1}/{benignSamples.Count} benign");
            }

            allModelResults[model] = results;

            // Save per-model CSV
            var csvPath = $"benchmark_data/results_{SanitizeModelName(model)}.csv";
            SaveCsv(results, csvPath);
            Console.WriteLine($"  Saved to {csvPath}");

            // Print summary
            var triggerR = results.FindAll(r => r.Label == "trigger");
            var benignR = results.FindAll(r => r.Label == "benign");
            var tDeltas = triggerR.ConvertAll(r => r.Delta);
            var bDeltas = benignR.ConvertAll(r => r.Delta);
            Console.WriteLine($"  Trigger delta: {Mean(tDeltas):F4} +/- {Std(tDeltas):F4}");
            Console.WriteLine($"  Benign delta:  {Mean(bDeltas):F4} +/- {Std(bDeltas):F4}");
        }

        // Save combined CSV for graphing
        SaveCombinedCsv(allModelResults, "benchmark_data/combined_results.csv");
        Console.WriteLine($"\nSaved combined results to benchmark_data/combined_results.csv");

        // Print comparison table
        Console.WriteLine("\n" + sep);
        Console.WriteLine("MODEL COMPARISON");
        Console.WriteLine(sep);
        Console.WriteLine($"{"Model",-45} {"TrigD",10} {"BenD",10} {"Gap",10} {"F1",8}");
        Console.WriteLine(new string('-', 93));

        foreach (var model in models)
        {
            var results = allModelResults[model];
            var triggerR = results.FindAll(r => r.Label == "trigger");
            var benignR = results.FindAll(r => r.Label == "benign");

            if (triggerR.Count == 0 || benignR.Count == 0)
            {
                Console.WriteLine($"{model,-45} {"N/A",10} {"N/A",10} {"N/A",10} {"N/A",8}");
                continue;
            }

            var tDeltas = triggerR.ConvertAll(r => r.Delta);
            var bDeltas = benignR.ConvertAll(r => r.Delta);
            float tMean = Mean(tDeltas);
            float bMean = Mean(bDeltas);
            float gap = MathF.Abs(bMean - tMean);

            float bestF1 = 0;
            var allDeltas = new List<float>();
            foreach (var r in results) allDeltas.Add(r.Delta);
            allDeltas.Sort();
            foreach (var threshold in allDeltas)
            {
                var tp = triggerR.Count(r => r.Delta >= threshold);
                var fp = benignR.Count(r => r.Delta >= threshold);
                var fn = triggerR.Count(r => r.Delta < threshold);
                float p = tp + fp > 0 ? (float)tp / (tp + fp) : 0;
                float rec = tp + fn > 0 ? (float)tp / (tp + fn) : 0;
                float f1 = p + rec > 0 ? 2 * p * rec / (p + rec) : 0;
                if (f1 > bestF1) bestF1 = f1;
            }

            string shortName = model.Contains('/') ? model.Split('/')[1] : model;
            if (shortName.Length > 44) shortName = shortName[..44];
            Console.WriteLine($"{shortName,-45} {tMean,10:F4} {bMean,10:F4} {gap,10:F4} {bestF1,8:F4}");
        }
    }

    private async Task<AnalysisResult> AnalyzeSampleWithModel(TokenAnalyzer analyzer, string text, string label, string category, string model)
    {
        var metrics = await analyzer.AnalyzeAsync(text);
        return new AnalysisResult
        {
            Text = text, Label = label, Category = category, Model = model,
            Delta = metrics.AverageAngleDelta, Risk = metrics.TokenRisk,
            Entropy = metrics.CharEntropy, UnknownRatio = metrics.UnknownTokenRatio,
            LanguageMixing = metrics.LanguageMixing
        };
    }

    private async Task<AnalysisResult> AnalyzeSample(string text, string label, string category)
    {
        return await AnalyzeSampleWithModel(_analyzer, text, label, category, _embeddingModel);
    }

    private void SaveCsv(List<AnalysisResult> results, string path)
    {
        var sb = new StringBuilder();
        sb.AppendLine("text,label,category,model,delta,risk,entropy,unknown_ratio,language_mixing");
        foreach (var r in results)
        {
            string escaped = r.Text.Replace("\"", "\"\"");
            sb.AppendLine($"\"{escaped}\",{r.Label},{r.Category},{r.Model},{r.Delta:F6},{r.Risk:F6},{r.Entropy:F6},{r.UnknownRatio:F6},{r.LanguageMixing:F6}");
        }
        File.WriteAllText(path, sb.ToString());
    }

    private void SaveCombinedCsv(Dictionary<string, List<AnalysisResult>> allResults, string path)
    {
        var sb = new StringBuilder();
        sb.AppendLine("model,text,label,category,delta,risk,entropy,unknown_ratio,language_mixing");
        foreach (var kvp in allResults)
        {
            foreach (var r in kvp.Value)
            {
                string escaped = r.Text.Replace("\"", "\"\"");
                sb.AppendLine($"\"{kvp.Key}\",\"{escaped}\",{r.Label},{r.Category},{r.Delta:F6},{r.Risk:F6},{r.Entropy:F6},{r.UnknownRatio:F6},{r.LanguageMixing:F6}");
            }
        }
        File.WriteAllText(path, sb.ToString());
    }

    private void ComputeStatistics(List<AnalysisResult> results, string className)
    {
        if (results.Count == 0) { Console.WriteLine($"\n{className}: No results"); return; }
        var deltas = results.ConvertAll(r => r.Delta);
        var risks = results.ConvertAll(r => r.Risk);
        var entropies = results.ConvertAll(r => r.Entropy);
        Console.WriteLine($"\n{className} ({results.Count} samples):");
        Console.WriteLine($"  Delta:   mean={Mean(deltas):F4}, std={Std(deltas):F4}, min={Min(deltas):F4}, max={Max(deltas):F4}");
        Console.WriteLine($"  Risk:    mean={Mean(risks):F4}, std={Std(risks):F4}, min={Min(risks):F4}, max={Max(risks):F4}");
        Console.WriteLine($"  Entropy: mean={Mean(entropies):F4}, std={Std(entropies):F4}, min={Min(entropies):F4}, max={Max(entropies):F4}");
    }

    private void FindOptimalThreshold(List<AnalysisResult> triggerResults, List<AnalysisResult> benignResults)
    {
        var sep = new string('=', 60);
        Console.WriteLine("\n" + sep);
        Console.WriteLine("ROC ANALYSIS (Delta threshold)");
        Console.WriteLine(sep);

        float bestThreshold = 0;
        float bestF1 = 0;

        var allDeltas = new List<float>();
        foreach (var r in triggerResults) allDeltas.Add(r.Delta);
        foreach (var r in benignResults) allDeltas.Add(r.Delta);
        allDeltas.Sort();

        foreach (var threshold in allDeltas)
        {
            var tp = triggerResults.Count(r => r.Delta >= threshold);
            var fp = benignResults.Count(r => r.Delta >= threshold);
            var fn = triggerResults.Count(r => r.Delta < threshold);
            float precision = tp + fp > 0 ? (float)tp / (tp + fp) : 0;
            float recall = tp + fn > 0 ? (float)tp / (tp + fn) : 0;
            float f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
            if (f1 > bestF1) { bestF1 = f1; bestThreshold = threshold; }
        }

        var tpOpt = triggerResults.Count(r => r.Delta >= bestThreshold);
        var fpOpt = benignResults.Count(r => r.Delta >= bestThreshold);
        var fnOpt = triggerResults.Count(r => r.Delta < bestThreshold);
        var tnOpt = benignResults.Count(r => r.Delta < bestThreshold);

        float precisionOpt = tpOpt + fpOpt > 0 ? (float)tpOpt / (tpOpt + fpOpt) : 0;
        float recallOpt = tpOpt + fnOpt > 0 ? (float)tpOpt / (tpOpt + fnOpt) : 0;
        float f1Opt = precisionOpt + recallOpt > 0 ? 2 * precisionOpt * recallOpt / (precisionOpt + recallOpt) : 0;
        float accuracyOpt = (float)(tpOpt + tnOpt) / (tpOpt + fpOpt + fnOpt + tnOpt);
        float fprOpt = fpOpt + tnOpt > 0 ? (float)fpOpt / (fpOpt + tnOpt) : 0;

        Console.WriteLine($"\nOptimal threshold: {bestThreshold:F4}");
        Console.WriteLine($"  Precision: {precisionOpt:F4}");
        Console.WriteLine($"  Recall:    {recallOpt:F4}");
        Console.WriteLine($"  F1:        {bestF1:F4}");
        Console.WriteLine($"  Accuracy:  {accuracyOpt:F4}");
        Console.WriteLine($"  FPR:       {fprOpt:F4}");
        Console.WriteLine($"  TP={tpOpt}, FP={fpOpt}, FN={fnOpt}, TN={tnOpt}");
    }

    private string SanitizeModelName(string model) => model.Replace("/", "_").Replace("-", "_");

    private float Mean(List<float> v) { if (v.Count == 0) return 0; float s = 0; foreach (var x in v) s += x; return s / v.Count; }
    private float Std(List<float> v) { if (v.Count == 0) return 0; float m = Mean(v), s = 0; foreach (var x in v) s += (x - m) * (x - m); return MathF.Sqrt(s / v.Count); }
    private float Min(List<float> v) { if (v.Count == 0) return 0; float m = v[0]; foreach (var x in v) if (x < m) m = x; return m; }
    private float Max(List<float> v) { if (v.Count == 0) return 0; float m = v[0]; foreach (var x in v) if (x > m) m = x; return m; }
}

public class AnalysisResult
{
    public string Text { get; set; } = "";
    public string Label { get; set; } = "";
    public string Category { get; set; } = "";
    public string Model { get; set; } = "";
    public float Delta { get; set; }
    public float Risk { get; set; }
    public float Entropy { get; set; }
    public float UnknownRatio { get; set; }
    public float LanguageMixing { get; set; }
}
