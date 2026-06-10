using System.Globalization;
using System.Text;

namespace AmDon;

public class TokenAnalyzer
{
    private readonly NimClient _nim;
    private readonly string _embeddingModel;

    public TokenAnalyzer(NimClient nim, string embeddingModel)
    {
        _nim = nim;
        _embeddingModel = embeddingModel;
    }

    public async Task<TokenMetrics> AnalyzeAsync(string input)
    {
        var embedding = await _nim.EmbedAsync(_embeddingModel, input, "passage") ?? [];

        var charEntropy = ComputeCharEntropy(input);
        var unknownTokenRatio = ComputeUnknownTokenRatio(input);
        var tokenBoundaryBreaks = ComputeTokenBoundaryBreaks(input);
        var repeatedPatterns = ComputeRepeatedPatterns(input);
        var languageMixing = ComputeLanguageMixing(input);

        var metrics = new TokenMetrics
        {
            Embedding = embedding,
            CharEntropy = charEntropy,
            UnknownTokenRatio = unknownTokenRatio,
            TokenBoundaryBreaks = tokenBoundaryBreaks,
            RepeatedPatterns = repeatedPatterns,
            LanguageMixing = languageMixing
        };

        metrics.AverageAngleDelta = ComputeAverageAngleDelta(embedding);
        metrics.TokenRisk = ComputeTokenRisk(metrics);

        return metrics;
    }

    private float ComputeCharEntropy(string input)
    {
        if (string.IsNullOrEmpty(input)) return 0f;

        var freq = new Dictionary<char, int>();
        foreach (var c in input)
        {
            if (!freq.ContainsKey(c)) freq[c] = 0;
            freq[c]++;
        }

        var entropy = 0f;
        var len = (float)input.Length;
        foreach (var count in freq.Values)
        {
            var p = count / len;
            if (p > 0) entropy -= p * MathF.Log2(p);
        }

        return entropy / MathF.Log2(Math.Min(freq.Count, 256));
    }

    private float ComputeUnknownTokenRatio(string input)
    {
        if (string.IsNullOrEmpty(input)) return 0f;

        var unusualChars = 0;
        foreach (var c in input)
        {
            if (c > 127 || char.IsControl(c)) unusualChars++;
        }

        return (float)unusualChars / input.Length;
    }

    private float ComputeTokenBoundaryBreaks(string input)
    {
        if (string.IsNullOrEmpty(input)) return 0f;

        var breaks = 0;
        var prevType = GetCharType(input[0]);

        for (int i = 1; i < input.Length; i++)
        {
            var currType = GetCharType(input[i]);
            if (currType != prevType) breaks++;
            prevType = currType;
        }

        return (float)breaks / input.Length;
    }

    private float ComputeRepeatedPatterns(string input)
    {
        if (string.IsNullOrEmpty(input) || input.Length < 4) return 0f;

        var patterns = new Dictionary<string, int>();
        for (int len = 2; len <= Math.Min(8, input.Length / 2); len++)
        {
            for (int i = 0; i <= input.Length - len; i++)
            {
                var pattern = input.Substring(i, len);
                if (!patterns.ContainsKey(pattern)) patterns[pattern] = 0;
                patterns[pattern]++;
            }
        }

        var repeated = patterns.Count(p => p.Value > 1);
        return (float)repeated / patterns.Count;
    }

    private float ComputeLanguageMixing(string input)
    {
        if (string.IsNullOrEmpty(input)) return 0f;

        var scripts = new HashSet<string>();
        foreach (var c in input)
        {
            var script = CharUnicodeInfo.GetUnicodeCategory(c).ToString();
            scripts.Add(script);
        }

        return Math.Min(1f, (float)(scripts.Count - 1) / 5);
    }

    private float ComputeAverageAngleDelta(float[] embedding)
    {
        if (embedding == null || embedding.Length < 2) return 0f;

        var angles = new List<float>();
        for (int i = 1; i < embedding.Length; i++)
        {
            var dot = embedding[i - 1] * embedding[i];
            var mag1 = MathF.Abs(embedding[i - 1]);
            var mag2 = MathF.Abs(embedding[i]);

            if (mag1 > 0 && mag2 > 0)
            {
                var cosAngle = Math.Clamp(dot / (mag1 * mag2), -1f, 1f);
                angles.Add(MathF.Acos(cosAngle));
            }
        }

        return angles.Count > 0 ? angles.Average() : 0f;
    }

    private float ComputeTokenRisk(TokenMetrics metrics)
    {
        var risk = 0f;
        risk += metrics.CharEntropy * 0.2f;
        risk += metrics.UnknownTokenRatio * 0.3f;
        risk += metrics.TokenBoundaryBreaks * 0.2f;
        risk += metrics.RepeatedPatterns * 0.15f;
        risk += metrics.LanguageMixing * 0.15f;
        return Math.Clamp(risk, 0f, 1f);
    }

    private int GetCharType(char c)
    {
        if (char.IsLetter(c)) return 0;
        if (char.IsDigit(c)) return 1;
        if (char.IsWhiteSpace(c)) return 2;
        return 3;
    }
}

public class TokenMetrics
{
    public float[] Embedding { get; set; } = [];
    public float CharEntropy { get; set; }
    public float UnknownTokenRatio { get; set; }
    public float TokenBoundaryBreaks { get; set; }
    public float RepeatedPatterns { get; set; }
    public float LanguageMixing { get; set; }
    public float AverageAngleDelta { get; set; }
    public float TokenRisk { get; set; }
}