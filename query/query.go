package query

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

const (
	CollectionName     = "rag-documents"
	EmbeddingDimension = 768
)

type Config struct {
	QdrantURL      string
	OllamaURL      string
	EmbeddingModel string
	AnthropicURL   string
	AnthropicToken string
	AnthropicModel string
	TopK           int
}

type Result struct {
	Answer  string
	Sources []string
}

// Query runs a one-shot RAG query and returns the answer with source references
func Query(ctx context.Context, cfg Config, question string) (*Result, error) {
	if cfg.TopK == 0 {
		cfg.TopK = 3
	}

	store, err := createVectorStore(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating vector store: %w", err)
	}

	docs, err := store.SimilaritySearch(ctx, question, cfg.TopK,
		vectorstores.WithScoreThreshold(0.5),
	)
	if err != nil {
		return nil, fmt.Errorf("similarity search: %w", err)
	}

	if len(docs) == 0 {
		return &Result{
			Answer: "I couldn't find any relevant information in the stored documents to answer your question.",
		}, nil
	}

	fmt.Printf("Found %d relevant document chunks\n", len(docs))

	var sources []string
	for _, doc := range docs {
		if src, ok := doc.Metadata["source"]; ok {
			sources = append(sources, fmt.Sprintf("%v", src))
		}
	}

	llm, err := createLLM(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating LLM: %w", err)
	}

	retriever := vectorstores.ToRetriever(store, cfg.TopK,
		vectorstores.WithScoreThreshold(0.5),
	)

	qaChain := chains.NewRetrievalQAFromLLM(llm, retriever)

	answer, err := chains.Run(ctx, qaChain, question)
	if err != nil {
		return nil, fmt.Errorf("running QA chain: %w", err)
	}

	return &Result{
		Answer:  answer,
		Sources: sources,
	}, nil
}

type Session struct {
	cfg    Config
	memory *memory.ConversationBuffer
	llm    *anthropic.LLM
}

// NewSession creates a new interactive query session with conversation memory
func NewSession(cfg Config) (*Session, error) {
	llm, err := createLLM(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating LLM: %w", err)
	}

	return &Session{
		cfg:    cfg,
		memory: memory.NewConversationBuffer(),
		llm:    llm,
	}, nil
}

// Query answers a question using conversation history and retrieved document context
func (s *Session) Query(ctx context.Context, question string) (*Result, error) {
	searchQuery, err := s.rewriteWithContext(ctx, question)
	if err != nil {
		searchQuery = question
	}

	cfg := s.cfg
	if cfg.TopK == 0 {
		cfg.TopK = 3
	}

	store, err := createVectorStore(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating vector store: %w", err)
	}

	docs, err := store.SimilaritySearch(ctx, searchQuery, cfg.TopK,
		vectorstores.WithScoreThreshold(0.5),
	)
	if err != nil {
		return nil, fmt.Errorf("similarity search: %w", err)
	}

	if len(docs) == 0 {
		answer := "I couldn't find any relevant information in the stored documents to answer your question."
		s.saveToMemory(ctx, question, answer)
		return &Result{Answer: answer}, nil
	}

	fmt.Printf("Found %d relevant document chunks\n", len(docs))

	var sources []string
	for _, doc := range docs {
		if src, ok := doc.Metadata["source"]; ok {
			sources = append(sources, fmt.Sprintf("%v", src))
		}
	}

	var contextParts []string
	for _, doc := range docs {
		contextParts = append(contextParts, doc.PageContent)
	}
	combinedContext := strings.Join(contextParts, "\n\n")

	history, _ := s.memory.LoadMemoryVariables(ctx, map[string]any{})
	historyStr, _ := history["history"].(string)

	prompt := fmt.Sprintf(`Use the following context to answer the question. If the conversation history is relevant, use it to understand the question better.

Conversation history:
%s

Context from documents:
%s

Question: %s

Answer:`, historyStr, combinedContext, question)

	answer, err := llms.GenerateFromSinglePrompt(ctx, s.llm, prompt)
	if err != nil {
		return nil, fmt.Errorf("generating answer: %w", err)
	}

	s.saveToMemory(ctx, question, answer)

	return &Result{
		Answer:  answer,
		Sources: sources,
	}, nil
}

// rewriteWithContext rewrites a follow-up question into a standalone query using conversation history
func (s *Session) rewriteWithContext(ctx context.Context, question string) (string, error) {
	history, err := s.memory.LoadMemoryVariables(ctx, map[string]any{})
	if err != nil {
		return question, err
	}

	historyStr, ok := history["history"].(string)
	if !ok || historyStr == "" {
		return question, nil
	}

	prompt := fmt.Sprintf(`Given the following conversation history and a new question, rewrite the question to be standalone and self-contained (so it can be used for a semantic search). Only output the rewritten question, nothing else.

Conversation history:
%s

New question: %s

Standalone question:`, historyStr, question)

	rewritten, err := llms.GenerateFromSinglePrompt(ctx, s.llm, prompt)
	if err != nil {
		return question, err
	}

	rewritten = strings.TrimSpace(rewritten)
	if rewritten == "" {
		return question, nil
	}

	fmt.Printf("Rewritten query: %s\n", rewritten)
	return rewritten, nil
}

// saveToMemory stores a question-answer pair in the conversation buffer
func (s *Session) saveToMemory(ctx context.Context, question, answer string) {
	_ = s.memory.SaveContext(ctx, map[string]any{"input": question}, map[string]any{"output": answer})
}

// SearchDocuments performs a similarity search and returns matching text chunks
func SearchDocuments(ctx context.Context, cfg Config, queryText string) ([]string, error) {
	if cfg.TopK == 0 {
		cfg.TopK = 5
	}

	store, err := createVectorStore(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating vector store: %w", err)
	}

	docs, err := store.SimilaritySearch(ctx, queryText, cfg.TopK,
		vectorstores.WithScoreThreshold(0.5),
	)
	if err != nil {
		return nil, fmt.Errorf("similarity search: %w", err)
	}

	var results []string
	for _, doc := range docs {
		results = append(results, doc.PageContent)
	}

	return results, nil
}

// createVectorStore initializes a Qdrant vector store with Ollama embeddings
func createVectorStore(cfg Config) (qdrant.Store, error) {
	ollamaModel := cfg.EmbeddingModel
	if ollamaModel == "" {
		ollamaModel = "nomic-embed-text"
	}

	opts := []ollama.Option{ollama.WithModel(ollamaModel)}
	if cfg.OllamaURL != "" {
		opts = append(opts, ollama.WithServerURL(cfg.OllamaURL))
	}

	llm, err := ollama.New(opts...)
	if err != nil {
		return qdrant.Store{}, fmt.Errorf("creating ollama client: %w", err)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		return qdrant.Store{}, fmt.Errorf("creating embedder: %w", err)
	}

	qdrantURL := cfg.QdrantURL
	if qdrantURL == "" {
		qdrantURL = "http://localhost:6333"
	}

	parsedURL, err := url.Parse(qdrantURL)
	if err != nil {
		return qdrant.Store{}, fmt.Errorf("parsing qdrant URL: %w", err)
	}

	if err := ensureCollection(qdrantURL, CollectionName, EmbeddingDimension); err != nil {
		return qdrant.Store{}, fmt.Errorf("ensuring collection: %w", err)
	}

	store, err := qdrant.New(
		qdrant.WithURL(*parsedURL),
		qdrant.WithCollectionName(CollectionName),
		qdrant.WithEmbedder(embedder),
	)
	if err != nil {
		return qdrant.Store{}, fmt.Errorf("creating qdrant store: %w", err)
	}

	return store, nil
}

// ensureCollection creates the Qdrant collection if it does not already exist.
func ensureCollection(qdrantURL, name string, dimension int) error {
	checkURL := fmt.Sprintf("%s/collections/%s", qdrantURL, name)
	resp, err := http.Get(checkURL) //nolint:gosec
	if err != nil {
		return fmt.Errorf("checking collection: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil
	}

	body := map[string]any{
		"vectors": map[string]any{
			"size":     dimension,
			"distance": "Cosine",
		},
	}
	b, _ := json.Marshal(body)

	createURL := fmt.Sprintf("%s/collections/%s", qdrantURL, name)
	req, err := http.NewRequest(http.MethodPut, createURL, bytes.NewReader(b))
	if err != nil {
		return fmt.Errorf("building create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("creating collection: %w", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		return fmt.Errorf("creating collection returned status %d", resp2.StatusCode)
	}
	return nil
}

// createLLM creates an Anthropic LLM client with the configured token and model
func createLLM(cfg Config) (*anthropic.LLM, error) {
	opts := []anthropic.Option{
		anthropic.WithToken(cfg.AnthropicToken),
	}
	if cfg.AnthropicURL != "" {
		opts = append(opts, anthropic.WithBaseURL(cfg.AnthropicURL))
	}
	if cfg.AnthropicModel != "" {
		opts = append(opts, anthropic.WithModel(cfg.AnthropicModel))
	}

	llm, err := anthropic.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("creating anthropic LLM: %w", err)
	}

	return llm, nil
}
