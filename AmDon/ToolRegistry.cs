namespace AmDon;

public class ToolRegistry
{
    private readonly Dictionary<string, ToolDefinition> _tools = new();

    public ToolRegistry()
    {
        // Meta tools — discover available tools
        Register(new ToolDefinition(
            "tool_search",
            "Search available tools by name or description. Returns matching tool names.",
            new Dictionary<string, string> { ["query"] = "Search query (matches name or description)" }
        ));

        Register(new ToolDefinition(
            "tool_details",
            "Get full details for a specific tool including parameters and usage.",
            new Dictionary<string, string> { ["tool_name"] = "Name of the tool to look up" }
        ));
    }

    public void Register(ToolDefinition tool)
    {
        _tools[tool.Name] = tool;
    }

    public ToolDefinition? Get(string name)
    {
        return _tools.TryGetValue(name, out var tool) ? tool : null;
    }

    public List<ToolDefinition> Search(string query)
    {
        var q = query.ToLower();
        return _tools.Values
            .Where(t => t.Name.Contains(q, StringComparison.OrdinalIgnoreCase) ||
                       t.Description.Contains(q, StringComparison.OrdinalIgnoreCase))
            .ToList();
    }

    public IReadOnlyDictionary<string, ToolDefinition> All => _tools;

    public string HandleMetaCall(ToolCall call)
    {
        return call.Name switch
        {
            "tool_search" => HandleToolSearch(call),
            "tool_details" => HandleToolDetails(call),
            _ => $"Unknown meta tool: {call.Name}"
        };
    }

    private string HandleToolSearch(ToolCall call)
    {
        if (!call.Parameters.TryGetValue("query", out var query))
            return "Missing required parameter: query";

        var results = Search(query);
        if (results.Count == 0)
            return $"No tools found matching '{query}'";

        return string.Join("\n", results.Select(t => $"- {t.Name}: {t.Description}"));
    }

    private string HandleToolDetails(ToolCall call)
    {
        if (!call.Parameters.TryGetValue("tool_name", out var name))
            return "Missing required parameter: tool_name";

        var tool = Get(name);
        if (tool == null)
            return $"Tool '{name}' not found";

        var paramLines = string.Join("\n", tool.Parameters.Select(p => $"  - {p.Key}: {p.Value}"));
        return $"""
            Tool: {tool.Name}
            Description: {tool.Description}
            Parameters:
            {paramLines}
            """;
    }
}
