package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"doc-qa/ingest"
	"doc-qa/query"
)

// main is the CLI entry point that routes commands to their handlers
func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cfg := loadConfig()

	fmt.Println("config loaded successfully", cfg)

	ctx := context.Background()

	switch os.Args[1] {
	case "ingest":
		fmt.Println("handling ingestion")
		handleIngest(ctx, cfg)
	case "query":
		handleQuery(ctx, cfg)
		fmt.Println("handling query")
	case "search":
		handleSearch(ctx, cfg)
		fmt.Println("handling search")
	default:
		printUsage()
		os.Exit(1)
	}
}

// handleIngest parses ingest subcommands and ingests files, directories, or raw text
func handleIngest(ctx context.Context, cfg appConfig) {
	if len(os.Args) < 3 {
		fmt.Println("Usage:")
		fmt.Println("  ai-rag ingest <file1> [file2] ...   - Ingest specific files")
		fmt.Println("  ai-rag ingest --dir <directory>     - Ingest all supported files in a directory")
		fmt.Println("  ai-rag ingest --text \"<text>\"        - Ingest raw text")
		os.Exit(1)
	}

	ingestCfg := ingest.Config{
		QdrantURL:      cfg.qdrantURL,
		OllamaURL:      cfg.ollamaURL,
		EmbeddingModel: cfg.embeddingModel,
		VisionModel:    cfg.visionModel,
	}

	switch os.Args[2] {
	case "--dir":
		if len(os.Args) < 4 {
			fmt.Println("Error: --dir requires a directory path")
			os.Exit(1)
		}
		if err := ingest.IngestDirectory(ctx, ingestCfg, os.Args[3]); err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
	case "--text":
		if len(os.Args) < 4 {
			fmt.Println("Error: --text requires text content")
			os.Exit(1)
		}
		text := strings.Join(os.Args[3:], " ")
		if err := ingest.IngestText(ctx, ingestCfg, text, "cli-input"); err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
	default:
		files := os.Args[2:]
		if err := ingest.IngestFiles(ctx, ingestCfg, files); err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
	}
}
// handleQuery starts a single question or interactive RAG query session
func handleQuery(ctx context.Context, cfg appConfig) {
	queryCfg := query.Config{
		QdrantURL:      cfg.qdrantURL,
		OllamaURL:      cfg.ollamaURL,
		EmbeddingModel: cfg.embeddingModel,
		AnthropicURL:   cfg.anthropicURL,
		AnthropicToken: cfg.anthropicToken,
		AnthropicModel: cfg.anthropicModel,
	}

	if len(os.Args) >= 3 {
		question := strings.Join(os.Args[2:], " ")
		askQuestion(ctx, queryCfg, question)
		return
	}

	session, err := query.NewSession(queryCfg)
	if err != nil {
		fmt.Printf("Error creating session: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("RAG Query Mode (with conversation memory)")
	fmt.Println("Follow-up questions will use context from previous answers.")
	fmt.Println("Type 'exit' to quit")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("Question: ")
		if !scanner.Scan() {
			break
		}
		question := strings.TrimSpace(scanner.Text())
		if question == "" {
			continue
		}
		if question == "exit" || question == "quit" {
			break
		}

		result, err := session.Query(ctx, question)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			fmt.Printf("\nAnswer: %s\n", result.Answer)
			if len(result.Sources) > 0 {
				unique := uniqueStrings(result.Sources)
				fmt.Printf("\nSources: %s\n", strings.Join(unique, ", "))
			}
		}
		fmt.Println()
	}
}
// handleSearch runs a similarity search and prints matching document chunks
func handleSearch(ctx context.Context, cfg appConfig) {
	if len(os.Args) < 3 {
		fmt.Println("Usage: ai-rag search <query>")
		os.Exit(1)
	}

	searchQuery := strings.Join(os.Args[2:], " ")

	queryCfg := query.Config{
		QdrantURL:      cfg.qdrantURL,
		OllamaURL:      cfg.ollamaURL,
		EmbeddingModel: cfg.embeddingModel,
	}

	results, err := query.SearchDocuments(ctx, queryCfg, searchQuery)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	if len(results) == 0 {
		fmt.Println("No relevant documents found.")
		return
	}

	fmt.Printf("Found %d relevant chunks:\n\n", len(results))
	for i, result := range results {
		fmt.Printf("--- Chunk %d ---\n%s\n\n", i+1, result)
	}
}

// askQuestion runs a one-shot RAG query and prints the answer with sources
func askQuestion(ctx context.Context, cfg query.Config, question string) {
	result, err := query.Query(ctx, cfg, question)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("\nAnswer: %s\n", result.Answer)
	if len(result.Sources) > 0 {
		unique := uniqueStrings(result.Sources)
		fmt.Printf("\nSources: %s\n", strings.Join(unique, ", "))
	}
}
