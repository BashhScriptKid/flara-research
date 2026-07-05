using System.Text.Json;

namespace AmDon;

public record MemoryEntry(
    string Id,
    string Content,
    float[] Embedding,
    DateTime Created,
    DateTime LastAccessed,
    int AccessCount,
    string Namespace
);

public class MemoryStore
{
    private readonly Dictionary<string, List<MemoryEntry>> _stores = new();
    private readonly int _maxEpisodic;
    private readonly string _persistPath;
    private readonly string _embeddingModel;
    public bool StoreEnabled { get; set; } = true;

    public MemoryStore(int maxEpisodic = 1000, string persistPath = "memory.json", string embeddingModel = "nvidia/nv-embedqa-e5-v5")
    {
        _maxEpisodic = maxEpisodic;
        _persistPath = persistPath;
        _embeddingModel = embeddingModel;
        _stores["room"] = new();           // Ephemeral scratchpad (bounded, cleared at session end)
        _stores["episodic"] = new();       // Interactions that passed promotion threshold
        _stores["semantic"] = new();       // Stable knowledge from consolidation
        _stores["belief"] = new();         // Belief system entries
        _stores["context_ledger"] = new(); // Guard decisions for pattern deduplication
        _stores["director_history"] = new();
        _stores["consolidation_candidates"] = new(); // Flagged for promotion to semantic
        Load();
    }

    public void Store(string ns, string content, float[] embedding)
    {
        if (!_stores.ContainsKey(ns))
            _stores[ns] = new();

        var entry = new MemoryEntry(
            Id: Guid.NewGuid().ToString("N")[..12],
            Content: content,
            Embedding: embedding,
            Created: DateTime.UtcNow,
            LastAccessed: DateTime.UtcNow,
            AccessCount: 0,
            Namespace: ns
        );

        _stores[ns].Add(entry);

        // Bounded Room namespace — max 50 entries, oldest removed first
        if (ns == "room" && _stores[ns].Count > 50)
            _stores[ns].RemoveAt(0);

        if (ns == "episodic" && _stores[ns].Count > _maxEpisodic)
            _stores[ns].RemoveAt(0);

        // Memory decay ladder
        if (ns == "episodic")
        {
            if (_stores[ns].Count > 500)
                CompressEntries(ns, 200);
            else if (_stores[ns].Count > 200)
                CompressEntries(ns, 50);
        }

        Save();
    }

    // Check if message is significant enough for episodic, or stays in Room
    // Uses model-based classification instead of heuristics
    public async Task<bool> IsSignificantEnoughForEpisodicAsync(string input, NimClient nim, string model)
    {
        // Delegate to model for significance classification
        var prompt = $"""
            Given this user message, is it significant enough to remember long-term?
            Significant: questions, opinions, emotional expressions, requests, personal info, decisions, creative ideas
            Trivial: greetings, acknowledgments, single-word responses, filler text

            User message: "{input}"

            Reply with ONLY: significant or trivial
            """;

        var resp = await nim.ChatAsync(model, new[] { new ChatMessage("user", prompt) }, maxTokens: 10);
        var result = resp?.Choices[0].Message.Content?.Trim().ToLower() ?? "trivial";

        return result.Contains("significant");
    }

    // Sleep cycle — model glances at episodic, builds linked tree, compacts by theme
    public async Task<ConsolidationResult> RunSleepCycleAsync(NimClient nim, string model)
    {
        if (!_stores.ContainsKey("episodic") || _stores["episodic"].Count == 0)
            return new ConsolidationResult(0, 0, "No episodic entries to consolidate");

        // 1. Gather all episodic entries
        var entries = _stores["episodic"].ToList();
        var entryList = string.Join("\n", entries.Select((e, i) => $"[{i}] {e.Id}: {e.Content}"));

        // 2. Gather existing semantic entries for linking
        var existingSemantic = _stores.ContainsKey("semantic")
            ? string.Join("\n", _stores["semantic"].Select(e => $"[{e.Id}] {e.Content.Split('\n')[0]}"))
            : "(none)";

        // 3. Model glances at everything and builds linked graph
        var treePrompt = $"""
            You are reviewing episodic memories to build a semantic knowledge graph.
            Analyze all entries and identify:

            1. ENTITIES — people, places, things, concepts (assign IDs like ent_A, ent_B)
            2. RELATIONSHIPS between entities (A friend_of B, A likes X, A owns Y)
            3. PROPERTIES — attributes, preferences, facts about each entity
            4. LINKS to existing semantic entries (if any overlap)

            Format your response as:
            ENTITY: ent_[id] — [name/description]
            - property: [what they like/are/know]
            - relationship: ent_[other] [relationship_type] ent_[other2]
            - links_to_semantic: [existing_semantic_id if this relates to something already stored]

            Relationships: [ent_A] [relationship] [ent_B]: [description]

            ---
            Existing semantic entries:
            {existingSemantic}

            New episodic entries:
            {entryList}
            """;

        var treeResp = await nim.ChatAsync(model, new[] { new ChatMessage("user", treePrompt) }, maxTokens: 1500);
        var tree = treeResp?.Choices[0].Message.Content ?? "";

        // 4. Model compacts each entity into lossless summary with links preserved
        var compactPrompt = $"""
            Given this knowledge graph structure, create compact semantic entries for each entity.
            Preserve all relationships and links — this is a linked graph, not a flat list.
            Each entry should know what it connects to.

            Format:
            ENTITY: [entity_id] — [name]
            SUMMARY: [compact summary preserving key facts]
            LINKS: [comma-separated entity_ids this connects to]
            SEMANTIC_LINKS: [comma-separated existing semantic_ids this relates to]
            ENTRIES_CONSUMED: [entry indices this was built from]

            ---
            Graph:
            {tree}
            """;

        var compactResp = await nim.ChatAsync(model, new[] { new ChatMessage("user", compactPrompt) }, maxTokens: 1500);
        var compacted = compactResp?.Choices[0].Message.Content ?? "";

        // 5. Parse and store semantic entries with links
        var entities = ParseCompactedEntities(compacted);
        var promotedCount = 0;

        foreach (var entity in entities)
        {
            var embedding = await nim.EmbedAsync(_embeddingModel, entity.Summary, "passage");
            if (embedding != null)
            {
                var semanticEntry = new MemoryEntry(
                    Id: Guid.NewGuid().ToString("N")[..12],
                    Content: $"ENTITY: {entity.Name}\n{entity.Summary}",
                    Embedding: embedding,
                    Created: DateTime.UtcNow,
                    LastAccessed: DateTime.UtcNow,
                    AccessCount: 0,
                    Namespace: "semantic"
                );
                _stores["semantic"].Add(semanticEntry);
                promotedCount++;

                // Store links for this entity
                _entityLinks[semanticEntry.Id] = new EntityLinks
                {
                    EntityId = semanticEntry.Id,
                    OutgoingLinks = entity.Links,
                    SemanticLinks = entity.SemanticLinks
                };

                // Remove consumed entries from episodic
                foreach (var entryIdx in entity.EntriesConsumed)
                {
                    if (entryIdx < entries.Count)
                    {
                        var entry = entries[entryIdx];
                        _stores["episodic"].Remove(entry);
                    }
                }
            }
        }

        // 6. Update inverse links (A links to B → B knows A links to it)
        UpdateInverseLinks();

        // 7. Run decay on remaining episodic
        if (_stores["episodic"].Count > 200)
            CompressEntries("episodic", 50);

        Save();
        return new ConsolidationResult(promotedCount, _stores["episodic"].Count, "Sleep cycle complete — graph built");
    }

    private List<ParsedEntity> ParseCompactedEntities(string compacted)
    {
        var entities = new List<ParsedEntity>();
        var blocks = compacted.Split("ENTITY:", StringSplitOptions.RemoveEmptyEntries);

        foreach (var block in blocks)
        {
            var lines = block.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            if (lines.Length < 2) continue;

            var name = lines[0].Trim();
            var summary = "";
            var links = new List<string>();
            var semanticLinks = new List<string>();
            var entriesConsumed = new List<int>();

            foreach (var line in lines.Skip(1))
            {
                if (line.StartsWith("SUMMARY:"))
                    summary = line["SUMMARY:".Length..].Trim();
                else if (line.StartsWith("LINKS:"))
                    links = line["LINKS:".Length..].Split(',', StringSplitOptions.RemoveEmptyEntries)
                        .Select(l => l.Trim()).ToList();
                else if (line.StartsWith("SEMANTIC_LINKS:"))
                    semanticLinks = line["SEMANTIC_LINKS:".Length..].Split(',', StringSplitOptions.RemoveEmptyEntries)
                        .Select(l => l.Trim()).ToList();
                else if (line.StartsWith("ENTRIES_CONSUMED:"))
                {
                    var indices = line["ENTRIES_CONSUMED:".Length..].Split(',', StringSplitOptions.RemoveEmptyEntries);
                    foreach (var idx in indices)
                    {
                        if (int.TryParse(idx.Trim(), out var i))
                            entriesConsumed.Add(i);
                    }
                }
            }

            if (!string.IsNullOrEmpty(summary))
                entities.Add(new ParsedEntity(name, summary, links, semanticLinks, entriesConsumed));
        }

        return entities;
    }

    private void UpdateInverseLinks()
    {
        foreach (var (id, links) in _entityLinks)
        {
            foreach (var targetId in links.OutgoingLinks)
            {
                if (_entityLinks.TryGetValue(targetId, out var targetLinks))
                {
                    if (!targetLinks.InverseLinks.Contains(id))
                        targetLinks.InverseLinks.Add(id);
                }
            }
        }
    }

    // Query semantic graph — find entities by relationship
    public List<(string EntityId, string Content, float Score)> QuerySemanticGraph(string query, float[] queryEmbedding, int topK = 5)
    {
        var results = Query("semantic", queryEmbedding, topK);

        // Enrich with link information
        return results.Select(r =>
        {
            var links = _entityLinks.TryGetValue(r.Entry.Id, out var l) ? l : null;
            var linkInfo = links != null
                ? $" [links: {string.Join(", ", links.OutgoingLinks)}]"
                : "";
            return (r.Entry.Id, r.Entry.Content + linkInfo, r.Score);
        }).ToList();
    }

    // Get all links for an entity
    public EntityLinks? GetEntityLinks(string entityId)
    {
        return _entityLinks.TryGetValue(entityId, out var links) ? links : null;
    }

    // Belief system hard-links to semantic entries
    public async Task<string> StoreBeliefWithLinksAsync(string key, string value, List<string> linkedSemanticIds)
    {
        var content = $"belief:{key}={value}";
        var id = await StoreAsync("belief", content);

        // Create hard links to semantic entries
        _beliefLinks[id] = linkedSemanticIds;

        // Mark linked semantic entries as belief-linked (decay suspended)
        foreach (var semanticId in linkedSemanticIds)
        {
            var entry = _stores["semantic"].FirstOrDefault(e => e.Id == semanticId);
            if (entry.Id != null)
            {
                // Update content to indicate belief link
                var updated = entry with
                {
                    Content = entry.Content + $" [BELIEF_LINKED: {id}]"
                };
                _stores["semantic"].Remove(entry);
                _stores["semantic"].Add(updated);
            }
        }

        Save();
        return id;
    }

    // Get semantic entries linked to a belief
    public List<MemoryEntry> GetBeliefLinkedSemantic(string beliefId)
    {
        if (!_beliefLinks.TryGetValue(beliefId, out var semanticIds))
            return new();

        return semanticIds
            .Select(id => _stores["semantic"].FirstOrDefault(e => e.Id == id))
            .Where(e => e.Id != null)
            .ToList();
    }

    public record ConsolidationResult(int PromotedToSemantic, int RemainingEpisodic, string Status);
    public record ParsedEntity(string Name, string Summary, List<string> Links, List<string> SemanticLinks, List<int> EntriesConsumed);

    // Graph structures
    private readonly Dictionary<string, EntityLinks> _entityLinks = new();
    private readonly Dictionary<string, List<string>> _beliefLinks = new();

    public class EntityLinks
    {
        public string EntityId { get; set; } = "";
        public List<string> OutgoingLinks { get; set; } = new();
        public List<string> InverseLinks { get; set; } = new();
        public List<string> SemanticLinks { get; set; } = new();
    }

    private void CompressEntries(string ns, int targetCount)
    {
        if (!_stores.ContainsKey(ns) || _stores[ns].Count <= targetCount)
            return;

        var entries = _stores[ns];
        var toCompress = entries.Take(entries.Count - targetCount).ToList();

        // Group similar entries by embedding similarity
        var groups = new List<List<MemoryEntry>>();
        foreach (var entry in toCompress)
        {
            var added = false;
            foreach (var group in groups)
            {
                if (group.Any(e => CosineSimilarity(e.Embedding, entry.Embedding) > 0.8f))
                {
                    group.Add(entry);
                    added = true;
                    break;
                }
            }
            if (!added)
                groups.Add(new List<MemoryEntry> { entry });
        }

        // Compress each group to a summary
        foreach (var group in groups)
        {
            var summary = CompressGroup(group);
            var summaryEntry = new MemoryEntry(
                Id: Guid.NewGuid().ToString("N")[..12],
                Content: summary,
                Embedding: group[0].Embedding,
                Created: group.Min(e => e.Created),
                LastAccessed: DateTime.UtcNow,
                AccessCount: group.Sum(e => e.AccessCount),
                Namespace: ns
            );

            // Remove original entries and add compressed one
            foreach (var entry in group)
                entries.Remove(entry);
            entries.Add(summaryEntry);
        }

        Save();
    }

    private string CompressGroup(List<MemoryEntry> group)
    {
        if (group.Count == 1)
            return group[0].Content;

        // Take key phrases from each entry
        var summaries = group.Select(e =>
        {
            var lines = e.Content.Split('\n');
            return lines.FirstOrDefault() ?? e.Content;
        }).ToList();

        return $"[{group.Count} entries] " + string.Join("; ", summaries);
    }

    public List<(MemoryEntry Entry, float Score)> Query(string ns, float[] queryEmbedding, int topK = 5)
    {
        if (!_stores.ContainsKey(ns))
            return new();

        return _stores[ns]
            .Select(e => (Entry: e, Score: CosineSimilarity(queryEmbedding, e.Embedding)))
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .ToList();
    }

    public List<(MemoryEntry Entry, float Score)> QueryAll(float[] queryEmbedding, int topK = 5)
    {
        return _stores.Values
            .SelectMany(e => e)
            .Select(e => (Entry: e, Score: CosineSimilarity(queryEmbedding, e.Embedding)))
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .ToList();
    }

    public int TotalEntries() => _stores.Values.Sum(s => s.Count);

    public void Clear(string? ns = null)
    {
        if (ns != null && _stores.ContainsKey(ns))
            _stores[ns].Clear();
        else if (ns == null)
            foreach (var key in _stores.Keys.ToList())
                _stores[key].Clear();
        Save();
    }

    public List<MemoryEntry> GetAll(string? ns = null)
    {
        if (ns != null && _stores.ContainsKey(ns))
            return _stores[ns].ToList();
        return _stores.Values.SelectMany(e => e).ToList();
    }

    public List<MemoryEntry> QueryNamespace(string ns, int limit = 20)
    {
        if (!_stores.ContainsKey(ns))
            return new();
        return _stores[ns].TakeLast(limit).Reverse().ToList();
    }

    // Belief system methods
    public async Task<string> StoreBeliefAsync(string key, string value)
    {
        var content = $"belief:{key}={value}";
        var id = await StoreAsync("belief", content);
        return id;
    }

    public string? GetBelief(string key)
    {
        if (!_stores.ContainsKey("belief"))
            return null;

        var entries = _stores["belief"];
        // Return most recent belief for this key
        var match = entries
            .Where(e => e.Content.StartsWith($"belief:{key}="))
            .OrderByDescending(e => e.Created)
            .FirstOrDefault();

        var prefix = $"belief:{key}=";
        return match?.Content[prefix.Length..];
    }

    public bool UpdateBelief(string key, string value)
    {
        if (!_stores.ContainsKey("belief"))
            return false;

        var entries = _stores["belief"];
        var existing = entries
            .Where(e => e.Content.StartsWith($"belief:{key}="))
            .ToList();

        // Remove old entries
        foreach (var entry in existing)
            entries.Remove(entry);

        // Store new
        var newEntry = new MemoryEntry(
            Id: Guid.NewGuid().ToString("N")[..12],
            Content: $"belief:{key}={value}",
            Embedding: [],
            Created: DateTime.UtcNow,
            LastAccessed: DateTime.UtcNow,
            AccessCount: 0,
            Namespace: "belief"
        );
        entries.Add(newEntry);
        Save();
        return true;
    }

    public List<(string Key, string Value)> GetAllBeliefs()
    {
        if (!_stores.ContainsKey("belief"))
            return new();

        return _stores["belief"]
            .Select(e =>
            {
                var content = e.Content;
                var eqIndex = content.IndexOf('=');
                if (eqIndex > 0 && content.StartsWith("belief:"))
                {
                    var key = content[7..eqIndex];
                    var value = content[(eqIndex + 1)..];
                    return (Key: key, Value: value);
                }
                return (Key: "", Value: content);
            })
            .Where(b => !string.IsNullOrEmpty(b.Key))
            .ToList();
    }

    public async Task<string> StoreAsync(string ns, string content)
    {
        var embedding = await Task.FromResult(new float[0]);
        Store(ns, content, embedding);
        return _stores[ns].Last().Id;
    }

    public bool Remove(string id)
    {
        foreach (var store in _stores.Values)
        {
            var entry = store.FirstOrDefault(e => e.Id == id);
            if (entry.Id != null)
            {
                store.Remove(entry);
                Save();
                return true;
            }
        }
        return false;
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length && i < b.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return normA == 0 || normB == 0 ? 0 : dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
    }

    private void Save()
    {
        var data = _stores.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.Select(e => new {
                e.Id, e.Content, e.Embedding, e.Created, e.LastAccessed, e.AccessCount, e.Namespace
            }).ToList()
        );

        var graphData = new
        {
            stores = data,
            entityLinks = _entityLinks,
            beliefLinks = _beliefLinks
        };

        File.WriteAllText(_persistPath, JsonSerializer.Serialize(graphData, new JsonSerializerOptions { WriteIndented = true }));
    }

    private void Load()
    {
        if (!File.Exists(_persistPath)) return;
        try
        {
            var json = File.ReadAllText(_persistPath);
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Load stores
            if (root.TryGetProperty("stores", out var storesElement))
            {
                var stores = JsonSerializer.Deserialize<Dictionary<string, List<MemoryEntry>>>(storesElement.GetRawText());
                if (stores != null)
                    foreach (var kvp in stores)
                        _stores[kvp.Key] = kvp.Value;
            }

            // Load entity links
            if (root.TryGetProperty("entityLinks", out var linksElement))
            {
                var links = JsonSerializer.Deserialize<Dictionary<string, EntityLinks>>(linksElement.GetRawText());
                if (links != null)
                    foreach (var kvp in links)
                        _entityLinks[kvp.Key] = kvp.Value;
            }

            // Load belief links
            if (root.TryGetProperty("beliefLinks", out var beliefElement))
            {
                var beliefs = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(beliefElement.GetRawText());
                if (beliefs != null)
                    foreach (var kvp in beliefs)
                        _beliefLinks[kvp.Key] = kvp.Value;
            }
        }
        catch { }
    }
}
