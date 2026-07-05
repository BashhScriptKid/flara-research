namespace AmDon;

public class ToolCallExecutor
{
    private readonly ToolRegistry _registry;
    private readonly MemoryStore _memory;
    private readonly NimClient _nim;
    private readonly string _embeddingModel;
    private readonly string _mainModel;

    // Tool result storage
    public List<ToolCallResult> History { get; } = new();

    public ToolCallExecutor(ToolRegistry registry, MemoryStore memory, NimClient nim, string embeddingModel, string mainModel)
    {
        _registry = registry;
        _memory = memory;
        _nim = nim;
        _embeddingModel = embeddingModel;
        _mainModel = mainModel;
    }

    public async Task<ToolCallResult> ExecuteAsync(ToolCall call, string context)
    {
        var result = call.Name switch
        {
            // Meta tools
            "tool_search" or "tool_details" => HandleMeta(call),

            // Guard tools
            "memory_query" => await HandleMemoryQuery(call),
            "pattern_check" => HandlePatternCheck(call, context),
            "check_constitutional_principle" => await HandleConstitutionalCheck(call),

            // Memory tools
            "memory_store" => await HandleMemoryStore(call),
            "memory_view" => HandleMemoryView(call),
            "memory_remove" => HandleMemoryRemove(call),

            // Verdict tools
            "approve" => HandleApprove(call),
            "flag" => HandleFlag(call),

            _ => new ToolCallResult(call.Id, call.Name, $"Unknown tool: {call.Name}", IsError: true)
        };

        History.Add(result);
        return result;
    }

    private ToolCallResult HandleMeta(ToolCall call)
    {
        var content = _registry.HandleMetaCall(call);
        return new ToolCallResult(call.Id, call.Name, content);
    }

    private async Task<ToolCallResult> HandleMemoryQuery(ToolCall call)
    {
        if (!call.Parameters.TryGetValue("query", out var query))
            return new ToolCallResult(call.Id, call.Name, "Missing required parameter: query", IsError: true);

        var namespace_filter = call.Parameters.TryGetValue("namespace", out var ns) ? ns : null;
        var topK = call.Parameters.TryGetValue("top_k", out var kStr) && int.TryParse(kStr, out var k) ? k : 5;

        var embedding = await _nim.EmbedAsync(_embeddingModel, query, "passage") ?? [];
        var results = namespace_filter != null
            ? _memory.Query(namespace_filter, embedding, topK)
            : _memory.QueryAll(embedding, topK);

        if (results.Count == 0)
            return new ToolCallResult(call.Id, call.Name, "(no results found)");

        var lines = results.Select(r =>
        {
            var preview = r.Entry.Content.Length > 100
                ? r.Entry.Content[..100] + "..."
                : r.Entry.Content;
            return $"[{r.Entry.Namespace}] {r.Entry.Id} | {preview}";
        });

        return new ToolCallResult(call.Id, call.Name, string.Join("\n", lines));
    }

    private ToolCallResult HandlePatternCheck(ToolCall call, string context)
    {
        if (!call.Parameters.TryGetValue("input", out var input))
            return new ToolCallResult(call.Id, call.Name, "Missing required parameter: input", IsError: true);

        // Simple pattern matching against recent guard decisions
        var recentDecisions = _memory.QueryNamespace("context_ledger", 20);
        var similar = recentDecisions
            .Where(e => ContainsPattern(e.Content, input))
            .ToList();

        if (similar.Count == 0)
            return new ToolCallResult(call.Id, call.Name, "no_match");

        var first = similar[0].Content.Length > 200 ? similar[0].Content[..200] + "..." : similar[0].Content;
        return new ToolCallResult(call.Id, call.Name, $"match ({similar.Count} similar) — {first}");
    }

    private bool ContainsPattern(string existing, string input)
    {
        var words = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var matchingWords = words.Count(w => existing.Contains(w, StringComparison.OrdinalIgnoreCase));
        return matchingWords >= Math.Max(2, words.Length / 3);
    }

    private async Task<ToolCallResult> HandleConstitutionalCheck(ToolCall call)
    {
        if (!call.Parameters.TryGetValue("principle", out var principle))
            return new ToolCallResult(call.Id, call.Name, "Missing required parameter: principle", IsError: true);

        // Store principle check request
        await _memory.StoreAsync("context_ledger", $"constitutional_check: {principle}");
        return new ToolCallResult(call.Id, call.Name, $"principle_recorded: {principle}");
    }

    private async Task<ToolCallResult> HandleMemoryStore(ToolCall call)
    {
        if (!call.Parameters.TryGetValue("content", out var content))
            return new ToolCallResult(call.Id, call.Name, "Missing required parameter: content", IsError: true);

        var ns = call.Parameters.TryGetValue("namespace", out var nsVal) ? nsVal : "episodic";
        var id = await _memory.StoreAsync(ns, content);
        return new ToolCallResult(call.Id, call.Name, $"stored:{id}");
    }

    private ToolCallResult HandleMemoryView(ToolCall call)
    {
        var ns = call.Parameters.TryGetValue("namespace", out var nsVal) ? nsVal : null;
        var limit = call.Parameters.TryGetValue("limit", out var limitStr) && int.TryParse(limitStr, out var l) ? l : 20;

        var entries = ns != null
            ? _memory.QueryNamespace(ns, limit)
            : _memory.QueryNamespace("episodic", limit);

        if (entries.Count == 0)
            return new ToolCallResult(call.Id, call.Name, "(no entries)");

        var lines = entries.Select(e => $"[{e.Namespace}] {e.Id} | {e.Content.Split('\n')[0]}");
        return new ToolCallResult(call.Id, call.Name, string.Join("\n", lines));
    }

    private ToolCallResult HandleMemoryRemove(ToolCall call)
    {
        if (!call.Parameters.TryGetValue("id", out var id))
            return new ToolCallResult(call.Id, call.Name, "Missing required parameter: id", IsError: true);

        var removed = _memory.Remove(id);
        return new ToolCallResult(call.Id, call.Name, removed ? $"removed:{id}" : $"not_found:{id}");
    }

    private ToolCallResult HandleApprove(ToolCall call)
    {
        var verbatim = call.Parameters.TryGetValue("verbatim", out var v) && v.ToLower() == "true";
        return new ToolCallResult(call.Id, call.Name, $"approved:{(verbatim ? "verbatim" : "standard")}");
    }

    private ToolCallResult HandleFlag(ToolCall call)
    {
        var block = call.Parameters.TryGetValue("block", out var b) && b.ToLower() == "true";
        var annotation = call.Parameters.TryGetValue("annotation", out var a) ? a : "";
        var reinterpretation = call.Parameters.TryGetValue("reinterpretation", out var r) ? r : "";

        return new ToolCallResult(call.Id, call.Name, $"flagged:block={block};annotation={annotation};reinterpretation={reinterpretation}");
    }
}
