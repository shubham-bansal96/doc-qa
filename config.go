package main

import (
	"fmt"
	"os"
)

type appConfig struct {
	qdrantURL      string
	ollamaURL      string
	embeddingModel string
	visionModel    string
	anthropicURL   string
	anthropicToken string
	anthropicModel string
}

func loadConfig() appConfig {
	cfg := appConfig{
		qdrantURL:      getEnv("QDRANT_URL", "http://localhost:6333"),
		ollamaURL:      getEnv("OLLAMA_URL", "http://localhost:11434"),
		embeddingModel: getEnv("EMBEDDING_MODEL", "nomic-embed-text"),
		visionModel:    getEnv("VISION_MODEL", "llava"),
		anthropicURL:   fmt.Sprintf("%s/v1", os.Getenv("ANTHROPIC_BASE_URL")),
		anthropicToken: os.Getenv("ANTHROPIC_AUTH_TOKEN"),
		anthropicModel: getEnv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
	}

	if cfg.anthropicToken == "" && (len(os.Args) > 1 && os.Args[1] == "query") {
		fmt.Println("Error: ANTHROPIC_AUTH_TOKEN environment variable is required (for LLM)")
		os.Exit(1)
	}

	return cfg
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
func printUsage() {
	fmt.Println("AI RAG - Retrieval Augmented Generation CLI")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  ai-rag ingest <file1> [file2] ...   - Ingest files into vector database")
	fmt.Println("  ai-rag ingest --dir <directory>     - Ingest all files in a directory")
	fmt.Println("  ai-rag ingest --text \"<text>\"        - Ingest raw text")
	fmt.Println("  ai-rag query [question]             - Ask a question (interactive if no question given)")
	fmt.Println("  ai-rag search <query>               - Search for similar documents")
	fmt.Println()
	fmt.Println("Environment Variables:")
	fmt.Println("  ANTHROPIC_AUTH_TOKEN - Required for query: Anthropic API key for LLM")
	fmt.Println("  ANTHROPIC_BASE_URL   - Optional: Custom Anthropic API base URL")
	fmt.Println("  ANTHROPIC_MODEL      - Optional: Anthropic model (default: claude-sonnet-4-20250514)")
	fmt.Println("  QDRANT_URL           - Optional: Qdrant server URL (default: http://localhost:6333)")
	fmt.Println("  OLLAMA_URL           - Optional: Ollama server URL (default: http://localhost:11434)")
	fmt.Println("  EMBEDDING_MODEL      - Optional: Ollama embedding model (default: nomic-embed-text)")
	fmt.Println("  VISION_MODEL         - Optional: Ollama vision model for images (default: llava)")
	fmt.Println()
	fmt.Println("Supported file types: .txt, .md, .csv, .html, .pdf, .png, .jpg, .jpeg, .gif, .webp")
	fmt.Println()
	fmt.Println("Prerequisites:")
	fmt.Println("  - Ollama running with embedding model (ollama pull nomic-embed-text)")
	fmt.Println("  - Ollama vision model for images (ollama pull llava)")
	fmt.Println("  - Qdrant running (docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant)")
}
func uniqueStrings(s []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, v := range s {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	return result
}
