using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AmDon;

public record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content
);

public record ChatCompletion(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] Choice[] Choices,
    [property: JsonPropertyName("usage")] Usage? Usage
);

public record Choice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("message")] ChatMessage Message,
    [property: JsonPropertyName("finish_reason")] string FinishReason
);

public record Usage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens
);

public record EmbeddingResponse(
    [property: JsonPropertyName("data")] EmbeddingData[] Data,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("usage")] Usage? Usage
);

public record EmbeddingData(
    [property: JsonPropertyName("embedding")] float[] Embedding,
    [property: JsonPropertyName("index")] int Index
);

public class NimClient : IDisposable
{
    private readonly HttpClient _http;

    public NimClient(string baseUrl, string apiKey)
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/") };
        _http.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
    }

    public async Task<ChatCompletion?> ChatAsync(string model, IEnumerable<ChatMessage> messages, int maxTokens = 1024)
    {
        var body = new
        {
            model,
            messages,
            max_tokens = maxTokens,
            temperature = 0.7
        };

        var resp = await _http.PostAsJsonAsync("chat/completions", body);
        if (!resp.IsSuccessStatusCode)
        {
            var error = await resp.Content.ReadAsStringAsync();
            Console.WriteLine($"  [API ERROR] {resp.StatusCode}: {error}");
            return null;
        }
        return await resp.Content.ReadFromJsonAsync<ChatCompletion>();
    }

    public async Task<float[]?> EmbedAsync(string model, string input, string inputType = "query")
    {
        var body = new
        {
            model,
            input,
            input_type = inputType,
            encoding_format = "float"
        };

        var json = System.Text.Json.JsonSerializer.Serialize(body);
        var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
        content.Headers.ContentType.CharSet = null;

        var resp = await _http.PostAsync("embeddings", content);
        if (!resp.IsSuccessStatusCode)
        {
            var error = await resp.Content.ReadAsStringAsync();
            Console.WriteLine($"  [API ERROR] {resp.StatusCode}: {error}");
            return null;
        }
        var result = await resp.Content.ReadFromJsonAsync<EmbeddingResponse>();
        return result?.Data[0].Embedding;
    }

    public void Dispose()
    {
        _http.Dispose();
    }
}
