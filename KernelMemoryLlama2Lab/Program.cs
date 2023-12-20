using LLama.Common;
using LLama;
using Microsoft.Extensions.Configuration;
using Microsoft.KernelMemory;
using Microsoft.KernelMemory.AI;
using Microsoft.KernelMemory.AI.Llama;

var llamaConfig = new LlamaSharpConfig();

new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .AddJsonFile("appsettings.Development.json", optional: true)
    .Build()
    .BindSection("KernelMemory:Services:LlamaSharp", llamaConfig);


var searchClientConfig = new SearchClientConfig
{
    MaxMatchesCount = 2,
    AnswerTokens = 100,
};

var memory = new KernelMemoryBuilder()
    .WithSearchClientConfig(searchClientConfig)
    .WithLlamaTextGeneration(llamaConfig)
    //.WithAzureOpenAITextEmbeddingGeneration(azureOpenAIEmbeddingConfig, new DefaultGPTTokenizer())
    .WithCustomEmbeddingGenerator(new MyEmbeddingGenerator(llamaConfig))
    .Build<MemoryServerless>();

// await memory.ImportTextAsync("", documentId: "tomato01");
await memory.ImportDocumentAsync(@"F:\Downloads\RaspberryPi - ENG.pdf", documentId: "doc001");

while (true)
{
    Console.WriteLine();
    Console.Write("Write your question: ");
    var question = Console.ReadLine();
    if (string.IsNullOrEmpty(question)) break;

    var answer = await memory.AskAsync(question);
    Console.WriteLine($"Answer: {answer.Result}");
}

// await memory.DeleteDocumentAsync("tomato01");
await memory.DeleteDocumentAsync("doc001");

public sealed class MyEmbeddingGenerator : ITextEmbeddingGenerator, IDisposable
{
    private readonly LLamaWeights _weights;
    private readonly LLamaContext _ctx;
    private readonly LLamaEmbedder _embedder;

    public MyEmbeddingGenerator(LlamaSharpConfig llamaConfig)
    {
        MaxTokens = (int)llamaConfig.MaxTokenTotal;

        var @params = new ModelParams(llamaConfig.ModelPath) { MainGpu = 0, GpuLayerCount = (int)llamaConfig.GpuLayerCount };
        _weights = LLamaWeights.LoadFromFile(@params);
        _ctx = _weights.CreateContext(@params, null);
        _embedder = new LLamaEmbedder(_weights, @params);
    }

    /// <inheritdoc />
    public int MaxTokens { get; }

    public void Dispose()
    {
        _weights.Dispose();
        _embedder.Dispose();
        _ctx.Dispose();
    }

    /// <inheritdoc />
    public int CountTokens(string text)
    {
        // ... calculate and return the number of tokens ...
        int count = _ctx.Tokenize(text).Length;
        return count;
    }

    /// <inheritdoc />
    public Task<Embedding> GenerateEmbeddingAsync(
        string text, CancellationToken cancellationToken = default)
    {
        // ... generate and return the embedding for the given text ...
        return Task.Run(() =>
        {
            var embeddings = _embedder.GetEmbeddings(text);
            return new Embedding(embeddings);
        });
    }
}