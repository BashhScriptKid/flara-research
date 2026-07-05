namespace AmDon;

public record ToolCall(
    string Id,
    string Name,
    Dictionary<string, string> Parameters
);

public record ToolCallResult(
    string ToolCallId,
    string Name,
    string Content,
    bool IsError = false
);

public record ToolDefinition(
    string Name,
    string Description,
    Dictionary<string, string> Parameters
);
