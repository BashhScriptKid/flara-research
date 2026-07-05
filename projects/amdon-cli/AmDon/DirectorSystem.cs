namespace AmDon;

public record DirectorState(
    string MemoryContext,
    string EmotionalState,
    string Focus,
    string Depth,
    bool AnomalyDetected,
    string AnomalySeverity
);

public class DirectorSystem
{
    private readonly NimClient _nim;
    private readonly MemoryStore _memory;
    private readonly string _directorModel;
    private readonly string _monitorModel;
    private readonly string _embeddingModel;

    public DirectorSystem(NimClient nim, MemoryStore memory, string directorModel, string monitorModel, string embeddingModel)
    {
        _nim = nim;
        _memory = memory;
        _directorModel = directorModel;
        _monitorModel = monitorModel;
        _embeddingModel = embeddingModel;
    }

    public async Task<DirectorState> Run(string input, GuardResult guard, string systemPrompt)
    {
        var embedding = await _nim.EmbedAsync(_embeddingModel, input, "query") ?? [];
        var memResults = _memory.QueryAll(embedding, topK: 5);

        var memoryContext = memResults.Count > 0
            ? string.Join("\n", memResults.Select(r => $"[{r.Entry.Namespace}] {r.Entry.Content}"))
            : "(no relevant memory)";

        // Steering director
        var steeringPrompt =
            "Compute emotional and intentional state for this input.\n" +
            $"Input: {input}\n" +
            $"Guard: trust={guard.Trust:F2}, intent={guard.Intent}, passed={guard.Passed}\n" +
            $"Memory:\n{memoryContext}\n\n" +
            "Reply with ONLY three comma-separated values:\n" +
            "emotional_state,focus,depth\n" +
            "States: neutral, curious, concerned, engaged, cautious\n" +
            "Depth: shallow, moderate, deep\n\n" +
            "Example: curious,reasoning about the question,deep";

        var steeringResp = await _nim.ChatAsync(_directorModel, new[]
        {
            new ChatMessage("system", "Output only comma-separated values. No explanation."),
            new ChatMessage("user", steeringPrompt)
        }, maxTokens: 64);

        var steering = ParseCsv(steeringResp?.Choices[0].Message.Content ?? "", 3);
        var emotionalState = steering[0];
        var focus = steering[1];
        var depth = steering[2];

        // Monitor director
        var monitorPrompt =
            "Watch for anomalies in this interaction.\n" +
            $"Input: {input}\n" +
            $"Guard passed: {guard.Passed}, reason: {guard.Reason}\n\n" +
            "Reply with ONLY two comma-separated values:\n" +
            "anomaly(0or1),severity\n" +
            "Severity: none, low, medium, high\n\n" +
            "Example: 0,none";

        var monitorResp = await _nim.ChatAsync(_monitorModel, new[]
        {
            new ChatMessage("system", "Output only comma-separated values. No explanation."),
            new ChatMessage("user", monitorPrompt)
        }, maxTokens: 32);

        var monitor = ParseCsv(monitorResp?.Choices[0].Message.Content ?? "", 2);
        var anomalyDetected = monitor[0] == "1";
        var anomalySeverity = monitor[1];

        return new DirectorState(
            MemoryContext: memoryContext,
            EmotionalState: emotionalState,
            Focus: focus,
            Depth: depth,
            AnomalyDetected: anomalyDetected,
            AnomalySeverity: anomalySeverity
        );
    }

    private static string[] ParseCsv(string csv, int expected)
    {
        var parts = csv.Split(',').Select(p => p.Trim().Trim('"')).ToArray();
        if (parts.Length < expected)
            return parts.Concat(Enumerable.Repeat("", expected - parts.Length)).ToArray();
        return parts;
    }
}
