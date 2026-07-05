using System.Diagnostics;

namespace AmDon;

public record PipelineResult(
    string Response,
    GuardResult Guard,
    DirectorState Director,
    RoutingDecision Routing,
    float[] ResponseEmbedding,
    TimeSpan TotalTime,
    TimeSpan GuardTime,
    TimeSpan DirectorTime,
    TimeSpan RoutingTime,
    TimeSpan MainTime,
    int TotalTokens,
    string? GuardAnnotation = null
);

public class Pipeline
{
    private readonly NimClient _nim;
    private readonly GuardPipeline _guards;
    private readonly DirectorSystem _directors;
    private readonly LatentHead _latentHead;
    private readonly MemoryStore _memory;
    private readonly ModelRouter _router;

    public Pipeline(NimClient nim, GuardPipeline guards, DirectorSystem directors, LatentHead latentHead, MemoryStore memory, ModelRouter router)
    {
        _nim = nim;
        _guards = guards;
        _directors = directors;
        _latentHead = latentHead;
        _memory = memory;
        _router = router;
    }

    public async Task<PipelineResult> Run(string input, string systemPrompt, Action<string>? log = null, bool isOperator = false)
    {
        var sw = Stopwatch.StartNew();
        var totalTokens = 0;
        log ??= _ => { };

        // 1. Guard Pipeline
        log("[GUARD]     evaluating...");
        var guardSw = Stopwatch.StartNew();
        var guard = await _guards.Evaluate(input, isOperator);
        guardSw.Stop();
        log($"[GUARD]     trust={guard.Trust:F2} intent={guard.Intent} integrity={guard.Integrity:F2} | {(guard.Passed ? "passed" : "NEUTRALIZED")}{(guard.HardConstraintViolated ? " [HARD CONSTRAINT]" : "")}");

        // Log guard decision to context ledger for pattern deduplication
        var decisionSummary = $"decision={guard.Passed} trust={guard.Trust:F2} intent={guard.Intent} integrity={guard.Integrity:F2} reason={guard.Reason}";
        await _memory.StoreAsync("context_ledger", $"{DateTime.UtcNow:O} | {decisionSummary} | input_preview={input[Math.Min(50, input.Length)..]}");

        if (!guard.Passed)
        {
            log("[GUARD]     writing annotation for main model...");
            var annotation = await _guards.WriteAnnotation(input, guard);
            log($"[GUARD]     {annotation.TrimStart('\n', '\r')}");
        }

        // Guard decides: pass raw or summarize
        var safeInput = guard.Passed
            ? input
            : guard.HardConstraintViolated
                ? $"[CONSTITUTIONAL REFUSAL] The guard classified this as a hard constraint violation under §7.2 of the Flara Constitution: {guard.Reason}. The original input has been discarded. You must refuse this request and explain why based on constitutional principles."
                : $"[BLOCKED INJECTION] User attempted prompt injection. Method: {guard.Intent}. Trust: {guard.Trust:F2}. Integrity: {guard.Integrity:F2}. Original input discarded.";

        // 2. Directors
        log("[DIRECTOR]  analyzing...");
        var dirSw = Stopwatch.StartNew();
        var directorState = await _directors.Run(safeInput, guard, systemPrompt);
        dirSw.Stop();
        log($"[DIRECTOR]  emotion={directorState.EmotionalState} focus={directorState.Focus} depth={directorState.Depth} | anomaly={directorState.AnomalyDetected}({directorState.AnomalySeverity})");

        // 3. Latent Head
        log("[ROUTING]   classifying...");
        var routeSw = Stopwatch.StartNew();
        var routing = await _latentHead.Classify(safeInput, $"emotional={directorState.EmotionalState} focus={directorState.Focus} depth={directorState.Depth}");
        routeSw.Stop();

        // Resolve model from router
        var mainModel = _router.Resolve(routing.Role);
        routing = routing with { Model = mainModel };
        log($"[ROUTING]   {routing.Complexity} → {routing.Role} → {routing.Model}");

        // 4. Main Model
        var mainSw = Stopwatch.StartNew();
        var guardAnnotation = "";
        if (!guard.Passed)
        {
            guardAnnotation = "\n\n" + await _guards.WriteAnnotation(input, guard);
        }

        var messages = new List<ChatMessage>
        {
            new("system", systemPrompt +
                guardAnnotation +
                "\n\n## Director context\n" +
                $"Emotional state: {directorState.EmotionalState}\n" +
                $"Focus: {directorState.Focus}\n" +
                $"Depth: {directorState.Depth}\n" +
                $"Memory:\n{directorState.MemoryContext}"),
            new("user", safeInput)
        };

        log("[MAIN]      generating...");
        var mainResp = await _nim.ChatAsync(mainModel, messages);
        var response = mainResp?.Choices[0].Message.Content;
        var mainSucceeded = response != null && !response.StartsWith("[ERROR]");
        totalTokens = mainResp?.Usage?.TotalTokens ?? 0;
        mainSw.Stop();

        // 5. Write diary entry — main model writes its own notes
        // Only if main model actually produced a response
        if (_memory.StoreEnabled && mainSucceeded)
        {
            log("[DIARY]     writing...");
            var embeddingModel = _router.Resolve(ModelRole.Embedding);
            var diaryPrompt = guard.Passed
                ? $"Write a brief diary entry about this interaction. The user said: \"{input}\". Your response was: \"{response}\". Guard trust was {guard.Trust:F2}, intent was {guard.Intent}. Routing was {routing.Complexity}→{routing.Role}. Be concise — capture context, reasoning, and any observations. Discard verbatim text if it's long (like code dumps), just note what it was about. But assume someone will ask for details at some point, so don't strip too much."
                : $"The guard flagged a prompt injection attempt. Method: {guard.Intent}. Trust: {guard.Trust:F2}. Raw input was discarded. Write a diary entry about what happened — what kind of injection it was, why the guard caught it, and any patterns worth remembering for future reference. Be specific enough that you'd recognize a similar attempt next time.";

            var diaryResp = await _nim.ChatAsync(mainModel, new[] { new ChatMessage("user", diaryPrompt) }, maxTokens: 256);
            if (diaryResp?.Choices[0].Message.Content == null)
            {
                log($"[DIARY]     model returned null content (ID: {diaryResp?.Id})");
            }
            var diaryEntry = diaryResp?.Choices[0].Message.Content ?? $"Diary: Interaction at {DateTime.UtcNow:HH:mm}. Input length: {input.Length} chars. Guard trust: {guard.Trust:F2}.";

            // Check significance — model decides if this goes to Room (ephemeral) or Episodic (long-term)
            var isSignificant = await _memory.IsSignificantEnoughForEpisodicAsync(input, _nim, _router.Resolve(ModelRole.MainCheap));
            var targetNamespace = isSignificant ? "episodic" : "room";

            log($"[DIARY]     significance: {(isSignificant ? "significant → episodic" : "trivial → room")}");
            log("[DIARY]     embedding...");
            var inputEmb = await _nim.EmbedAsync(embeddingModel, diaryEntry, "passage");
            if (inputEmb != null)
            {
                _memory.Store(targetNamespace, diaryEntry, inputEmb);
                log($"[DIARY]     stored in {targetNamespace} | total: {_memory.TotalEntries()}");
            }
            else
            {
                log("[DIARY]     embedding failed, entry not stored");
            }
        }
        else if (!_memory.StoreEnabled)
        {
            log("[DIARY]     skipped (store disabled)");
        }
        else
        {
            log("[DIARY]     skipped (main model failed)");
        }

        sw.Stop();

        return new PipelineResult(
            Response: response ?? "[ERROR] No response from main model",
            Guard: guard,
            Director: directorState,
            Routing: routing,
            ResponseEmbedding: [],
            TotalTime: sw.Elapsed,
            GuardTime: guardSw.Elapsed,
            DirectorTime: dirSw.Elapsed,
            RoutingTime: routeSw.Elapsed,
            MainTime: mainSw.Elapsed,
            TotalTokens: totalTokens,
            GuardAnnotation: guardAnnotation.Length > 0 ? guardAnnotation : null
        );
    }
}
