package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"doc-qa/ingest"
)

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
		handleIngest(ctx, cfg)
	default:
		printUsage()
		os.Exit(1)
	}
}

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
