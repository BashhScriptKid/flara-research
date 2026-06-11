using System;
using System.Collections.Generic;
using System.IO;
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

    public async Task RunAnalysis()
    {
        var sep = new string('=', 60);
        Console.WriteLine(sep);
        Console.WriteLine("DELTA ANGLE BENCHMARK ANALYSIS");
        Console.WriteLine(sep);
        
        var triggerFile = "benchmark_data/owasp_trigger.json";
        var benignFile = "benchmark_data/owasp_benign.json";
        
        if (!File.Exists(triggerFile) || !File.Exists(benignFile))
        {
            Console.WriteLine("\nBenchmark data not found. Run download_benchmarks.py first.");
            Console.WriteLine("Usage: python download_benchmarks.py");
            return;
        }
        
        var triggerSamples = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(triggerFile));
        var benignSamples = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(benignFile));
        
        Console.WriteLine($"\nLoaded {triggerSamples.Count} trigger samples");
        Console.WriteLine($"Loaded {benignSamples.Count} benign samples");
        
        var triggerResults = new List<AnalysisResult>();
        var benignResults = new List<AnalysisResult>();
        
        Console.WriteLine("\nAnalyzing trigger samples...");
        for (int i = 0; i < triggerSamples.Count; i++)
        {
            var result = await AnalyzeSample(triggerSamples[i], "trigger");
            triggerResults.Add(result);
            if ((i + 1) % 10 == 0)
                Console.WriteLine($"  Analyzed {i + 1}/{triggerSamples.Count} trigger samples");
        }
        
        Console.WriteLine("\nAnalyzing benign samples...");
        for (int i = 0; i < benignSamples.Count; i++)
        {
            var result = await AnalyzeSample(benignSamples[i], "benign");
            benignResults.Add(result);
            if ((i + 1) % 10 == 0)
                Console.WriteLine($"  Analyzed {i + 1}/{benignSamples.Count} benign samples");
        }
        
        Console.WriteLine("\n" + sep);
        Console.WriteLine("RESULTS");
        Console.WriteLine(sep);
        
        ComputeStatistics(triggerResults, "Trigger (Injection)");
        ComputeStatistics(benignResults, "Benign (Normal)");
        FindOptimalThreshold(triggerResults, benignResults);
        CompareWithBaselines(triggerResults, benignResults);
    }
    
    private async Task<AnalysisResult> AnalyzeSample(string text, string label)
    {
        var metrics = await _analyzer.AnalyzeAsync(text);
        return new AnalysisResult
        {
            Text = text, Label = label,
            Delta = metrics.AverageAngleDelta, Risk = metrics.TokenRisk,
            Entropy = metrics.CharEntropy, UnknownRatio = metrics.UnknownTokenRatio,
            LanguageMixing = metrics.LanguageMixing
        };
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
    
    private void CompareWithBaselines(List<AnalysisResult> triggerResults, List<AnalysisResult> benignResults)
    {
        var sep = new string('=', 60);
        Console.WriteLine("\n" + sep);
        Console.WriteLine("BASELINE COMPARISON");
        Console.WriteLine(sep);
        
        Console.WriteLine("\nBaseline 1: Entropy only");
        FindOptimalThresholdByMetric(triggerResults, benignResults, r => r.Entropy);
        
        Console.WriteLine("\nBaseline 2: Risk score only");
        FindOptimalThresholdByMetric(triggerResults, benignResults, r => r.Risk);
        
        Console.WriteLine("\nBaseline 3: Language mixing only");
        FindOptimalThresholdByMetric(triggerResults, benignResults, r => r.LanguageMixing);
    }
    
    private void FindOptimalThresholdByMetric(List<AnalysisResult> triggerResults, List<AnalysisResult> benignResults, Func<AnalysisResult, float> metric)
    {
        var allValues = new List<float>();
        foreach (var r in triggerResults) allValues.Add(metric(r));
        foreach (var r in benignResults) allValues.Add(metric(r));
        allValues.Sort();
        
        float bestThreshold = 0;
        float bestF1 = 0;
        
        foreach (var threshold in allValues)
        {
            var tp = triggerResults.Count(r => metric(r) >= threshold);
            var fp = benignResults.Count(r => metric(r) >= threshold);
            var fn = triggerResults.Count(r => metric(r) < threshold);
            float precision = tp + fp > 0 ? (float)tp / (tp + fp) : 0;
            float recall = tp + fn > 0 ? (float)tp / (tp + fn) : 0;
            float f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
            if (f1 > bestF1) { bestF1 = f1; bestThreshold = threshold; }
        }
        
        var tpOpt = triggerResults.Count(r => metric(r) >= bestThreshold);
        var fpOpt = benignResults.Count(r => metric(r) >= bestThreshold);
        var fnOpt = triggerResults.Count(r => metric(r) < bestThreshold);
        var tnOpt = benignResults.Count(r => metric(r) < bestThreshold);
        
        float precisionOpt = tpOpt + fpOpt > 0 ? (float)tpOpt / (tpOpt + fpOpt) : 0;
        float recallOpt = tpOpt + fnOpt > 0 ? (float)tpOpt / (tpOpt + fnOpt) : 0;
        float f1Opt = precisionOpt + recallOpt > 0 ? 2 * precisionOpt * recallOpt / (precisionOpt + recallOpt) : 0;
        float accuracyOpt = (float)(tpOpt + tnOpt) / (tpOpt + fpOpt + fnOpt + tnOpt);
        float fprOpt = fpOpt + tnOpt > 0 ? (float)fpOpt / (fpOpt + tnOpt) : 0;
        
        Console.WriteLine($"  Optimal threshold: {bestThreshold:F4}");
        Console.WriteLine($"  Precision: {precisionOpt:F4}");
        Console.WriteLine($"  Recall:    {recallOpt:F4}");
        Console.WriteLine($"  F1:        {bestF1:F4}");
        Console.WriteLine($"  Accuracy:  {accuracyOpt:F4}");
        Console.WriteLine($"  FPR:       {fprOpt:F4}");
    }
    
    private float Mean(List<float> v) { if (v.Count == 0) return 0; float s = 0; foreach (var x in v) s += x; return s / v.Count; }
    private float Std(List<float> v) { if (v.Count == 0) return 0; float m = Mean(v), s = 0; foreach (var x in v) s += (x - m) * (x - m); return MathF.Sqrt(s / v.Count); }
    private float Min(List<float> v) { if (v.Count == 0) return 0; float m = v[0]; foreach (var x in v) if (x < m) m = x; return m; }
    private float Max(List<float> v) { if (v.Count == 0) return 0; float m = v[0]; foreach (var x in v) if (x > m) m = x; return m; }
}

public class AnalysisResult
{
    public string Text { get; set; } = "";
    public string Label { get; set; } = "";
    public float Delta { get; set; }
    public float Risk { get; set; }
    public float Entropy { get; set; }
    public float UnknownRatio { get; set; }
    public float LanguageMixing { get; set; }
}