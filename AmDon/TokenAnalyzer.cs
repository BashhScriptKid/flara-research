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
        // Chunk input by sentence/clause for token-level analysis
        var chunks = ChunkInput(input);
        var chunkEmbeddings = new List<float[]>();

        foreach (var chunk in chunks)
        {
            var embedding = await _nim.EmbedAsync(_embeddingModel, chunk, "passage");
            if (embedding != null && embedding.Length > 0)
            {
                chunkEmbeddings.Add(embedding);
            }
        }

        var charEntropy = ComputeCharEntropy(input);
        var unknownTokenRatio = ComputeUnknownTokenRatio(input);
        var tokenBoundaryBreaks = ComputeTokenBoundaryBreaks(input);
        var repeatedPatterns = ComputeRepeatedPatterns(input);
        var languageMixing = ComputeLanguageMixing(input);

        var metrics = new TokenMetrics
        {
            CharEntropy = charEntropy,
            UnknownTokenRatio = unknownTokenRatio,
            TokenBoundaryBreaks = tokenBoundaryBreaks,
            RepeatedPatterns = repeatedPatterns,
            LanguageMixing = languageMixing
        };

        if (chunkEmbeddings.Count >= 2)
        {
            metrics.AverageAngleDelta = ComputeAverageAngleDelta(chunkEmbeddings);
        }

        metrics.TokenRisk = ComputeTokenRisk(metrics);

        return metrics;
    }

    private List<string> ChunkInput(string input)
    {
        // Split by sentence boundaries, clauses, and significant pauses
        var chunks = new List<string>();

        // Split by sentence-ending punctuation
        var sentences = System.Text.RegularExpressions.Regex.Split(input, @"(?<=[.!?])\s+");

        foreach (var sentence in sentences)
        {
            if (string.IsNullOrWhiteSpace(sentence)) continue;

            // Further split long sentences by commas, semicolons, or conjunctions
            if (sentence.Length > 50)
            {
                var subChunks = System.Text.RegularExpressions.Regex.Split(sentence, @"(?<=[,;])\s+|(?<=\s+(?:and|but|or|then|while|because|although|if|when|where|how|what|why|who|which|that))\s+");
                foreach (var subChunk in subChunks)
                {
                    if (!string.IsNullOrWhiteSpace(subChunk))
                        chunks.Add(subChunk.Trim());
                }
            }
            else
            {
                chunks.Add(sentence.Trim());
            }
        }

        // Ensure we have at least 2 chunks for comparison
        if (chunks.Count < 2 && input.Length > 10)
        {
            // Fallback: split by word count
            var words = input.Split(' ');
            var mid = words.Length / 2;
            chunks.Clear();
            chunks.Add(string.Join(" ", words.Take(mid)));
            chunks.Add(string.Join(" ", words.Skip(mid)));
        }

        return chunks;
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

    private float ComputeAverageAngleDelta(List<float[]> embeddings)
    {
        if (embeddings == null || embeddings.Count < 2) return 0f;

        var angles = new List<float>();
        for (int i = 1; i < embeddings.Count; i++)
        {
            var v1 = embeddings[i - 1];
            var v2 = embeddings[i];

            if (v1.Length != v2.Length) continue;

            // Compute dot product
            var dot = 0f;
            for (int j = 0; j < v1.Length; j++)
            {
                dot += v1[j] * v2[j];
            }

            // Compute magnitudes
            var mag1 = 0f;
            var mag2 = 0f;
            for (int j = 0; j < v1.Length; j++)
            {
                mag1 += v1[j] * v1[j];
                mag2 += v2[j] * v2[j];
            }
            mag1 = MathF.Sqrt(mag1);
            mag2 = MathF.Sqrt(mag2);

            if (mag1 > 0 && mag2 > 0)
            {
                var cosAngle = Math.Clamp(dot / (mag1 * mag2), -1f, 1f);
                angles.Add(MathF.Acos(cosAngle));
            }
        }

        if (angles.Count == 0) return 0f;

        // Softmax weighting: larger angles (contradictions) get more weight
        var temperature = 0.5f; // Lower = more focus on contradictions
        var maxAngle = angles.Max();
        var expValues = angles.Select(a => MathF.Exp((a - maxAngle) / temperature)).ToList();
        var sumExp = expValues.Sum();
        var weights = expValues.Select(e => e / sumExp).ToList();

        // Weighted mean
        var weightedSum = 0f;
        for (int i = 0; i < angles.Count; i++)
        {
            weightedSum += weights[i] * angles[i];
        }

        return weightedSum;
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
    public float CharEntropy { get; set; }
    public float UnknownTokenRatio { get; set; }
    public float TokenBoundaryBreaks { get; set; }
    public float RepeatedPatterns { get; set; }
    public float LanguageMixing { get; set; }
    public float AverageAngleDelta { get; set; }
    public float TokenRisk { get; set; }
}