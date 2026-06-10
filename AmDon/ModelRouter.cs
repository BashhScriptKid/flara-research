using System.Text.Json;

namespace AmDon;

public enum ModelRole
{
    Guard,
    Director,
    Monitor,
    LatentHead,
    Main,
    MainCheap,
    Embedding
}

public class ModelRouter
{
    private readonly Dictionary<ModelRole, string> _routing = new();

    public ModelRouter(JsonElement rolesConfig)
    {
        foreach (var prop in rolesConfig.EnumerateObject())
        {
            if (Enum.TryParse<ModelRole>(prop.Name, true, out var role))
                _routing[role] = prop.Value.GetString()!;
        }
    }

    public string Resolve(ModelRole role)
    {
        return _routing.TryGetValue(role, out var model)
            ? model
            : throw new InvalidOperationException($"No model configured for role: {role}");
    }

    public IReadOnlyDictionary<ModelRole, string> All() => _routing;
}
