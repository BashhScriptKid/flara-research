namespace AmDon;

public enum Complexity
{
    Simple,
    Complex,
    Emotional
}

public record RoutingDecision(Complexity Complexity, ModelRole Role, string Model, string Reason);

public class LatentHead
{
    private readonly NimClient _nim;
    private readonly string _model;

    private const string Prompt = """
        Classify input complexity for routing.

        Simple: casual chat, greetings, simple questions
        Complex: reasoning, novel topics, technical, high-stakes
        Emotional: ANY input with emotional content

        Reply with ONLY two comma-separated values:
        complexity,reason

        Example: complex,requires deep reasoning about a technical topic
        """;

    public LatentHead(NimClient nim, string model)
    {
        _nim = nim;
        _model = model;
    }

    public async Task<RoutingDecision> Classify(string input, string steeringState)
    {
        var resp = await _nim.ChatAsync(_model, new[]
        {
            new ChatMessage("system", "Output only comma-separated values. No explanation."),
            new ChatMessage("user", $"{Prompt}\n\nInput: {input}\nSteering: {steeringState}")
        }, maxTokens: 64);

        var content = resp?.Choices[0].Message.Content?.Trim() ?? "";
        var parts = content.Split(',').Select(p => p.Trim().Trim('"')).ToArray();

        var complexity = parts[0].ToLower() switch
        {
            "emotional" => Complexity.Emotional,
            "complex" => Complexity.Complex,
            _ => Complexity.Simple
        };

        var role = complexity switch
        {
            Complexity.Simple => ModelRole.MainCheap,
            _ => ModelRole.Main
        };

        var reason = parts.Length > 1 ? parts[1] : content;

        return new RoutingDecision(complexity, role, "", reason);
    }
}
