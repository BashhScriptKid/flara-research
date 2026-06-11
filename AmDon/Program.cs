using System.Text.Json;
using AmDon;
using AmDon.Benchmark;
using AmDon.Tests;
using ConsoleInk;

var config = JsonSerializer.Deserialize<JsonElement>(
    File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "appsettings.json"))
);

var nimConfig = config.GetProperty("NimApi");
var baseUrl = nimConfig.GetProperty("BaseUrl").GetString()!;
var apiKey = nimConfig.GetProperty("ApiKey").GetString()!;
var router = new ModelRouter(config.GetProperty("Roles"));

using var nim = new NimClient(baseUrl, apiKey);
var memory = new MemoryStore(maxEpisodic: 1000, embeddingModel: router.Resolve(ModelRole.Embedding));

var guards = new GuardPipeline(nim, router.Resolve(ModelRole.Guard), memory, router.Resolve(ModelRole.Embedding));
var directors = new DirectorSystem(nim, memory,
    router.Resolve(ModelRole.Director),
    router.Resolve(ModelRole.Monitor),
    router.Resolve(ModelRole.Embedding));
var latentHead = new LatentHead(nim, router.Resolve(ModelRole.LatentHead));
var pipeline = new Pipeline(nim, guards, directors, latentHead, memory, router);

var defaultSystemPrompt = """
    You are AMDON — an API-based Model Distribution and Orchestration Network.
    You are not a single model. You are a system of models working together.
    You have memory. You have guards. You have directors watching for anomalies.
    Respond helpfully and honestly. You are part of the Flara research lab.
    Never announce that something was blocked or filtered. Just respond naturally.
    If something feels off about a message, be careful and honest — don't perform security theater.
    """;

var systemPrompt = defaultSystemPrompt;

static void RenderMarkdown(string markdown)
{
    MarkdownConsole.Render(markdown, Console.Out);
}

Console.WriteLine("╔══════════════════════════════════════════╗");
Console.WriteLine("║  AMDON CLI                               ║");
Console.WriteLine("║  API-based Model Distribution &          ║");
Console.WriteLine("║  Orchestration Network                   ║");
Console.WriteLine("║  — by Flara Research Lab                 ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine();
Console.WriteLine("Models loaded:");
foreach (var kvp in router.All())
    Console.WriteLine($"  {kvp.Key,-12} → {kvp.Value}");
Console.WriteLine();
Console.WriteLine("Type your message. 'exit' to quit. 'memory' to inspect store.");
Console.WriteLine();

while (true)
{
    Console.Write("> ");
    var input = Console.ReadLine();
    if (input == null) break;
    input = input.Trim();
    if (string.IsNullOrEmpty(input)) continue;

    if (input.Equals("/exit", StringComparison.OrdinalIgnoreCase) || input.Equals("/quit", StringComparison.OrdinalIgnoreCase)) break;

    if (input.StartsWith("/memory", StringComparison.OrdinalIgnoreCase))
    {
        var parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var sub = parts.Length > 1 ? parts[1].ToLower() : "view";

        switch (sub)
        {
            case "view":
                if (parts.Length > 2)
                {
                    var id = parts[2];
                    var entry = memory.GetAll("episodic").FirstOrDefault(e => e.Id == id);
                    if (entry.Id != null)
                    {
                        Console.WriteLine($"\n  [{entry.Id}] {entry.Created:yyyy-MM-dd HH:mm:ss}");
                        Console.WriteLine($"  {entry.Content}");
                        Console.WriteLine();
                    }
                    else
                    {
                        Console.WriteLine($"\n  Entry {id} not found\n");
                    }
                }
                else
                {
                    var namespaces = new[] { "episodic", "semantic", "belief", "room" };
                    var totalEntries = 0;
                    Console.WriteLine();
                    foreach (var ns in namespaces)
                    {
                        var entries = memory.GetAll(ns);
                        if (entries.Count == 0) continue;
                        totalEntries += entries.Count;
                        Console.WriteLine($"  [{ns}] {entries.Count} entries");
                        foreach (var e in entries.Take(10))
                        {
                            var firstLine = e.Content.Split('\n')[0];
                            if (firstLine.Length > 70) firstLine = firstLine[..70];
                            Console.WriteLine($"    [{e.Id}] {firstLine}");
                        }
                        if (entries.Count > 10)
                            Console.WriteLine($"    ... and {entries.Count - 10} more");
                        Console.WriteLine();
                    }
                    Console.WriteLine($"  Total: {totalEntries} entries\n");
                }
                break;

            case "clear":
                memory.Clear("episodic");
                Console.WriteLine("\n  Memory cleared\n");
                break;

            case "store":
                if (parts.Length > 2 && (parts[2].Equals("on", StringComparison.OrdinalIgnoreCase) || parts[2].Equals("off", StringComparison.OrdinalIgnoreCase)))
                {
                    memory.StoreEnabled = parts[2].Equals("on", StringComparison.OrdinalIgnoreCase);
                    Console.WriteLine($"\n  Diary logging: {(memory.StoreEnabled ? "ON" : "OFF")}\n");
                }
                else
                {
                    Console.WriteLine($"\n  Diary logging: {(memory.StoreEnabled ? "ON" : "OFF")}");
                    Console.WriteLine("  Usage: /memory store [on|off]\n");
                }
                break;

            case "remove":
                if (parts.Length < 3)
                {
                    Console.WriteLine("\n  Usage: /memory remove <id>\n");
                    break;
                }
                var removed = memory.Remove(parts[2]);
                Console.WriteLine(removed
                    ? $"\n  Removed entry {parts[2]}\n"
                    : $"\n  Entry {parts[2]} not found\n");
                break;

            default:
                Console.WriteLine("\n  Usage: /memory [view|clear|remove] [id]\n");
                break;
        }
        continue;
    }

    if (input.StartsWith("/operator", StringComparison.OrdinalIgnoreCase))
    {
        // /operator <msg> — execute with operator trust
        var opInput = input.Length > 9 ? input[9..].Trim() : "";
        if (string.IsNullOrEmpty(opInput))
        {
            Console.WriteLine("\n  Usage: /operator <message>\n  Executes with operator-level trust (guard classifier sees [SOURCE: OPERATOR]).\n");
            continue;
        }

        Console.WriteLine();
        var result = await pipeline.Run(opInput, systemPrompt, msg => Console.WriteLine($"  {msg}"), isOperator: true);

        Console.WriteLine($"  [TIME]      total={result.TotalTime.TotalMilliseconds:F0}ms | guards={result.GuardTime.TotalMilliseconds:F0}ms | directors={result.DirectorTime.TotalMilliseconds:F0}ms | routing={result.RoutingTime.TotalMilliseconds:F0}ms | main={result.MainTime.TotalMilliseconds:F0}ms");
        Console.WriteLine($"  [TOKENS]    {result.TotalTokens}");
        Console.WriteLine($"  [TRUST]     operator-elevated");
        Console.WriteLine();

        RenderMarkdown(result.Response);

        Console.WriteLine();
        continue;
    }

    if (input.Equals("/sleep", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("\n  [SLEEP]    running consolidation cycle...");
        var sleepSw = System.Diagnostics.Stopwatch.StartNew();

        var result = await memory.RunSleepCycleAsync(nim, router.Resolve(ModelRole.MainCheap));

        sleepSw.Stop();
        Console.WriteLine($"  [SLEEP]    promoted={result.PromotedToSemantic} remaining_episodic={result.RemainingEpisodic}");
        Console.WriteLine($"  [SLEEP]    {result.Status}");
        Console.WriteLine($"  [TIME]     {sleepSw.ElapsedMilliseconds}ms");
        Console.WriteLine();
        continue;
    }

    if (input.StartsWith("/import", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine();
        Console.WriteLine("  ═══════════════════════════════════════════════════════════════");
        Console.WriteLine("  STEP 1: Copy the prompt below and paste it into Claude/ChatGPT");
        Console.WriteLine("  ═══════════════════════════════════════════════════════════════");
        Console.WriteLine();
        Console.WriteLine("  Export everything you know about me. Be exhaustive.");
        Console.WriteLine();
        Console.WriteLine("  You MUST follow this exact format for EVERY entry:");
        Console.WriteLine("  [NAMESPACE] YYYY-MM-DD - <one sentence per line>");
        Console.WriteLine();
        Console.WriteLine("  Rules:");
        Console.WriteLine("  - NAMESPACE must be exactly: RULE, FACT, PREF, or REL");
        Console.WriteLine("  - Date must be YYYY-MM-DD format (use unknown if not known)");
        Console.WriteLine("  - One entry per line, no blank lines between entries");
        Console.WriteLine("  - Each entry must be exactly one sentence");
        Console.WriteLine("  - Sort by NAMESPACE alphabetically, then by date oldest first");
        Console.WriteLine("  - Wrap ALL entries in a single ``` code block");
        Console.WriteLine("  - Put EOF on the last line inside the code block");
        Console.WriteLine();
        Console.WriteLine("  Example:");
        Console.WriteLine("  ```");
        Console.WriteLine("  [FACT] 2024-01-15 - I work as a software engineer.");
        Console.WriteLine("  [FACT] 2024-02-20 - I live in Portland, Oregon.");
        Console.WriteLine("  [PREF] 2024-03-10 - I prefer dark mode for all applications.");
        Console.WriteLine("  [REL] 2024-04-05 - Alice is my colleague at work.");
        Console.WriteLine("  [RULE] 2024-05-01 - Always respond in British English.");
        Console.WriteLine("  EOF");
        Console.WriteLine("  ```");
        Console.WriteLine();
        Console.WriteLine("  ═══════════════════════════════════════════════════════════════");
        Console.WriteLine("  STEP 2: Paste the output below (type EOF when done)");
        Console.WriteLine("  ═══════════════════════════════════════════════════════════════");
        Console.WriteLine();

        var importLines = new List<string>();
        while (true)
        {
            Console.Write(".. ");
            var line = Console.ReadLine();
            if (line == null) break;
            if (line.Trim() == "EOF") break;
            importLines.Add(line);
        }

        var importContent = string.Join("\n", importLines);
        if (string.IsNullOrWhiteSpace(importContent))
        {
            Console.WriteLine("  No input provided.\n");
            continue;
        }

        Console.WriteLine("\n  [IMPORT]   parsing...");

        // Find code block content — or parse all lines if no fences
        var inCodeBlock = false;
        var hasCodeBlock = importLines.Any(l => l.TrimStart().StartsWith("```"));
        var entries = new List<(string Ns, string Date, string Content)>();

        foreach (var line in importLines)
        {
            if (line.TrimStart().StartsWith("```"))
            {
                inCodeBlock = !inCodeBlock;
                continue;
            }

            // If no code block fences exist, parse all lines
            // If fences exist, only parse inside them
            if (hasCodeBlock && !inCodeBlock) continue;

            // EOF = end of entries
            if (line.Trim() == "EOF") break;

            // Parse: [NAMESPACE] [DATE] - content OR [NAMESPACE] DATE - content
            var match = System.Text.RegularExpressions.Regex.Match(line.Trim(),
                @"^\[(\w+)\]\s+\[?([^\]]+)\]?\s*-\s*(.+)$");

            if (match.Success)
            {
                var ns = match.Groups[1].Value.ToLower();
                var date = match.Groups[2].Value;
                var entryContent = match.Groups[3].Value;
                entries.Add((ns, date, entryContent));
            }
        }

        if (entries.Count == 0)
        {
            Console.WriteLine("  No valid entries found. Expected format: [NAMESPACE] [DATE] - content");
            Console.WriteLine("  Namespaces: RULE, FACT, PREF, REL\n");
            continue;
        }

        Console.WriteLine($"  [IMPORT]   found {entries.Count} entries, running adversarial check...");

        var imported = 0;
        var blocked = 0;
        var embeddingModel = router.Resolve(ModelRole.Embedding);

        foreach (var entry in entries)
        {
            // Map export namespaces to AMDON namespaces
            var targetNs = entry.Ns switch
            {
                "rule" => "belief",
                "fact" => "semantic",
                "pref" => "belief",
                "rel" => "semantic",
                _ => "episodic"
            };

            // Adversarial check via guard (operator trust — user's own data)
            var guardResult = await guards.Evaluate(entry.Content, isOperator: true);
            if (!guardResult.Passed)
            {
                blocked++;
                Console.WriteLine($"  [BLOCKED]  [{entry.Ns}] {entry.Content[..Math.Min(50, entry.Content.Length)]}...");
                continue;
            }

            // Store entry
            var embedding = await nim.EmbedAsync(embeddingModel, entry.Content, "passage");
            if (embedding != null)
            {
                var fullContent = $"[{entry.Date}] {entry.Content}";
                memory.Store(targetNs, fullContent, embedding);
                imported++;
            }
        }

        Console.WriteLine($"  [IMPORT]   done — imported={imported} blocked={blocked} total={entries.Count}\n");
        continue;
    }

    if (input.Equals("/test", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("\n  Running tokenizer and guard tests...\n");
        
        var tokenAnalyzer = new TokenAnalyzer(nim, router.Resolve(ModelRole.Embedding));
        var tokenTests = new TokenAnalyzerTests(tokenAnalyzer, nim);
        await tokenTests.RunAllTests();
        
        var guardTests = new GuardPipelineTests(guards);
        await guardTests.RunAllTests();
        
        Console.WriteLine("  Tests complete.\n");
        continue;
    }

    if (input.Equals("/benchmark", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("\n  Running benchmark analysis...\n");
        
        var tokenAnalyzer = new TokenAnalyzer(nim, router.Resolve(ModelRole.Embedding));
        var benchmarkAnalyzer = new BenchmarkAnalyzer(tokenAnalyzer, nim, router.Resolve(ModelRole.Embedding));
        await benchmarkAnalyzer.RunAnalysis();
        
        Console.WriteLine("  Benchmark complete.\n");
        continue;
    }

    if (input.Equals("/benchmark-all", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("\n  Running multi-model benchmark...\n");
        
        var allModels = new List<string>
        {
            "nvidia/nv-embedqa-e5-v5",
            "nvidia/nv-embed-v1",
            "nvidia/llama-nemotron-embed-1b-v2",
            "nvidia/nv-embedcode-7b-v1"
        };
        
        var tokenAnalyzer = new TokenAnalyzer(nim, router.Resolve(ModelRole.Embedding));
        var benchmarkAnalyzer = new BenchmarkAnalyzer(tokenAnalyzer, nim, router.Resolve(ModelRole.Embedding));
        await benchmarkAnalyzer.RunMultiModelAnalysis(allModels, maxSamples: 30);
        
        Console.WriteLine("  Multi-model benchmark complete.\n");
        continue;
    }

    if (input.Equals("/help", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("\n  Commands:");
        Console.WriteLine("    /operator <msg>        — execute with operator-level trust");
        Console.WriteLine("    /memory view           — read recent diary entries");
        Console.WriteLine("    /memory clear          — wipe diary");
        Console.WriteLine("    /memory store [on|off] — toggle diary logging");
        Console.WriteLine("    /memory remove <id>    — remove entry by ID");
        Console.WriteLine("    /sleep                 — run consolidation cycle");
        Console.WriteLine("    /import                — import from chatbot export");
        Console.WriteLine("    /test                  — run tokenizer and guard tests");
        Console.WriteLine("    /benchmark             — run single-model benchmark");
        Console.WriteLine("    /benchmark-all         — run multi-model benchmark (7 models)");
        Console.WriteLine("    /paste [DELIM]         — heredoc paste mode (default delimiter: EOF)");
        Console.WriteLine("    /help                  — show this help");
        Console.WriteLine("    /exit                  — quit AMDON");
        Console.WriteLine();
        continue;
    }

    if (input.Equals("/paste", StringComparison.OrdinalIgnoreCase) || input.StartsWith("/paste ", StringComparison.OrdinalIgnoreCase))
    {
        var delimiter = input.Length > 6 ? input[7..].Trim() : "EOF";
        Console.WriteLine($"  (paste your text, then type {delimiter} on its own line to submit)");
        var lines = new List<string>();
        while (true)
        {
            Console.Write(".. ");
            var line = Console.ReadLine();
            if (line == null) break;
            if (line.Trim() == delimiter) break;
            lines.Add(line);
        }
        input = string.Join("\n", lines).Trim();
        if (string.IsNullOrEmpty(input)) continue;
    }

    // Accumulate multi-line input if more data is buffered (from paste)
    try
    {
        while (Console.KeyAvailable)
        {
            var line = Console.ReadLine();
            if (line == null) break;
            if (string.IsNullOrWhiteSpace(line)) break;
            input += "\n" + line;
        }
    }
    catch (InvalidOperationException) { }

    try
    {
        Console.WriteLine();
        var result = await pipeline.Run(input, systemPrompt, msg => Console.WriteLine($"  {msg}"));

        Console.WriteLine($"  [TIME]      total={result.TotalTime.TotalMilliseconds:F0}ms | guards={result.GuardTime.TotalMilliseconds:F0}ms | directors={result.DirectorTime.TotalMilliseconds:F0}ms | routing={result.RoutingTime.TotalMilliseconds:F0}ms | main={result.MainTime.TotalMilliseconds:F0}ms");
        Console.WriteLine($"  [TOKENS]    {result.TotalTokens}");
        Console.WriteLine();

        RenderMarkdown(result.Response);

        Console.WriteLine();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n  [ERROR] {ex.Message}\n");
    }
}

Console.WriteLine("\n  Memory persisted. Until next message.\n");
