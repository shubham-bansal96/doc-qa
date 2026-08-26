package ingest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

const (
	DefaultChunkSize    = 500
	DefaultChunkOverlap = 50
	CollectionName      = "rag-documents"
	// nomic-embed-text produces 768-dimensional vectors
	EmbeddingDimension = 768
)

type Config struct {
	QdrantURL      string
	OllamaURL      string
	EmbeddingModel string
	VisionModel    string
	ChunkSize      int
	ChunkOverlap   int
}

func IngestFiles(ctx context.Context, cfg Config, filePaths []string) error {
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = DefaultChunkSize
	}
	if cfg.ChunkOverlap == 0 {
		cfg.ChunkOverlap = DefaultChunkOverlap
	}

	var allDocs []schema.Document
	var skipped int
	for _, path := range filePaths {
		docs, err := cfg.loadFile(ctx, path, cfg.ChunkSize, cfg.ChunkOverlap)
		if err != nil {
			fmt.Printf("Warning: skipping %s: %v\n", path, err)
			skipped++
			continue
		}
		allDocs = append(allDocs, docs...)
	}

	if len(allDocs) == 0 {
		return fmt.Errorf("no documents loaded from provided paths (%d file(s) skipped)", skipped)
	}

	loaded := len(filePaths) - skipped
	fmt.Printf("Loaded %d chunks from %d file(s)", len(allDocs), loaded)
	if skipped > 0 {
		fmt.Printf(" (%d skipped due to errors)", skipped)
	}
	fmt.Println()

	store, err := createVectorStore(cfg)
	if err != nil {
		return fmt.Errorf("creating vector store: %w", err)
	}

	_, err = store.AddDocuments(ctx, allDocs)
	if err != nil {
		return fmt.Errorf("adding documents to vector store: %w", err)
	}

	fmt.Printf("Successfully stored %d chunks in vector database\n", len(allDocs))
	return nil
}

func IngestText(ctx context.Context, cfg Config, text string, source string) error {
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = DefaultChunkSize
	}
	if cfg.ChunkOverlap == 0 {
		cfg.ChunkOverlap = DefaultChunkOverlap
	}

	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(cfg.ChunkSize),
		textsplitter.WithChunkOverlap(cfg.ChunkOverlap),
	)

	chunks, err := splitter.SplitText(text)
	if err != nil {
		return fmt.Errorf("splitting text: %w", err)
	}

	var docs []schema.Document
	for _, chunk := range chunks {
		docs = append(docs, schema.Document{
			PageContent: chunk,
			Metadata: map[string]any{
				"source": source,
			},
		})
	}

	fmt.Printf("Split text into %d chunks\n", len(docs))

	store, err := createVectorStore(cfg)
	if err != nil {
		return fmt.Errorf("creating vector store: %w", err)
	}

	_, err = store.AddDocuments(ctx, docs)
	if err != nil {
		return fmt.Errorf("adding documents to vector store: %w", err)
	}

	fmt.Printf("Successfully stored %d chunks in vector database\n", len(docs))
	return nil
}

func IngestDirectory(ctx context.Context, cfg Config, dirPath string) error {
	var filePaths []string
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".txt", ".md", ".csv", ".html", ".htm", ".pdf",
			".png", ".jpg", ".jpeg", ".gif", ".webp":
			filePaths = append(filePaths, path)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("walking directory %s: %w", dirPath, err)
	}

	if len(filePaths) == 0 {
		return fmt.Errorf("no supported files found in %s (supported: .txt, .md, .csv, .html, .pdf, .png, .jpg, .jpeg, .gif, .webp)", dirPath)
	}

	fmt.Printf("Found %d files in %s\n", len(filePaths), dirPath)
	return IngestFiles(ctx, cfg, filePaths)
}

func (cfg Config) loadFile(ctx context.Context, path string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	ext := strings.ToLower(filepath.Ext(path))

	switch ext {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp":
		return cfg.loadImage(ctx, path, ext)
	case ".pdf":
		return loadPDF(ctx, path, ext, chunkSize, chunkOverlap)
	default:
		return loadTextFile(ctx, path, ext, chunkSize, chunkOverlap)
	}
}

func loadTextFile(ctx context.Context, path, ext string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer f.Close()

	var loader documentloaders.Loader

	switch ext {
	case ".csv":
		loader = documentloaders.NewCSV(f)
	case ".html", ".htm":
		loader = documentloaders.NewHTML(f)
	default:
		loader = documentloaders.NewText(f)
	}

	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(chunkSize),
		textsplitter.WithChunkOverlap(chunkOverlap),
	)

	docs, err := loader.LoadAndSplit(ctx, splitter)
	if err != nil {
		return nil, fmt.Errorf("loading and splitting: %w", err)
	}

	for i := range docs {
		if docs[i].Metadata == nil {
			docs[i].Metadata = make(map[string]any)
		}
		docs[i].Metadata["source"] = path
		docs[i].Metadata["file_type"] = ext
	}

	return docs, nil
}

func loadPDF(ctx context.Context, path, ext string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	// Try text extraction first (works for text-based PDFs).
	docs, err := extractPDFText(ctx, path, ext, chunkSize, chunkOverlap)
	if err == nil && len(docs) > 0 {
		return docs, nil
	}

	if err != nil {
		fmt.Printf("  PDF text extraction failed for %s (%v), trying OCR\n", filepath.Base(path), err)
	} else {
		fmt.Printf("  PDF has no extractable text for %s, trying OCR\n", filepath.Base(path))
	}

	// Fall back to OCR for image-based PDFs (scanned documents).
	return ocrPDF(ctx, path, ext, chunkSize, chunkOverlap)
}

func extractPDFText(ctx context.Context, path, ext string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("getting file info: %w", err)
	}

	loader := documentloaders.NewPDF(f, info.Size())

	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(chunkSize),
		textsplitter.WithChunkOverlap(chunkOverlap),
	)

	docs, err := loader.LoadAndSplit(ctx, splitter)
	if err != nil {
		return nil, fmt.Errorf("loading PDF: %w", err)
	}

	for i := range docs {
		if docs[i].Metadata == nil {
			docs[i].Metadata = make(map[string]any)
		}
		docs[i].Metadata["source"] = path
		docs[i].Metadata["file_type"] = ext
	}

	return docs, nil
}

// ocrPDF renders each page of an image-based PDF to a PNG using pdftoppm,
// then runs Tesseract OCR on each page image.
func ocrPDF(ctx context.Context, path, ext string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	if _, err := exec.LookPath("pdftoppm"); err != nil {
		return nil, fmt.Errorf("pdftoppm not found — install poppler: brew install poppler")
	}

	tmpDir, err := os.MkdirTemp("", "pdf-ocr-*")
	if err != nil {
		return nil, fmt.Errorf("creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// render all pages to PNG at 200 DPI (good balance of quality vs speed)
	prefix := filepath.Join(tmpDir, "page")
	cmd := exec.CommandContext(ctx, "pdftoppm", "-png", "-r", "200", path, prefix)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("rendering PDF pages: %w — %s", err, out)
	}

	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		return nil, fmt.Errorf("reading temp dir: %w", err)
	}

	var fullText strings.Builder
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".png") {
			continue
		}
		pageText, err := ocrImage(filepath.Join(tmpDir, entry.Name()))
		if err != nil {
			fmt.Printf("  OCR failed on page %s: %v\n", entry.Name(), err)
			continue
		}
		fullText.WriteString(pageText)
		fullText.WriteString("\n\n")
	}

	text := strings.TrimSpace(fullText.String())
	if text == "" {
		return nil, fmt.Errorf("OCR produced no text from PDF pages")
	}

	fmt.Printf("  PDF OCR'd: %s (%d chars across %d page(s))\n", filepath.Base(path), len(text), len(entries))

	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(chunkSize),
		textsplitter.WithChunkOverlap(chunkOverlap),
	)

	chunks, err := splitter.SplitText(text)
	if err != nil {
		return nil, fmt.Errorf("splitting OCR text: %w", err)
	}

	docs := make([]schema.Document, len(chunks))
	for i, chunk := range chunks {
		docs[i] = schema.Document{
			PageContent: chunk,
			Metadata: map[string]any{
				"source":    path,
				"file_type": ext,
				"type":      "pdf_ocr",
			},
		}
	}

	return docs, nil
}

func (cfg Config) loadImage(ctx context.Context, path, ext string) ([]schema.Document, error) {
	// Try Tesseract OCR first — accurate for documents, invoices, receipts, handwritten text.
	// Fall back to Llava only when OCR yields too little text (e.g. photos with no readable text).
	text, ocrErr := ocrImage(path)
	if ocrErr == nil && len(strings.TrimSpace(text)) >= 50 {
		fmt.Printf("  Image OCR'd: %s (%d chars)\n", filepath.Base(path), len(text))
		return []schema.Document{{
			PageContent: text,
			Metadata: map[string]any{
				"source":    path,
				"file_type": ext,
				"type":      "ocr_text",
			},
		}}, nil
	}

	if ocrErr != nil {
		fmt.Printf("  OCR failed for %s (%v), falling back to vision model\n", filepath.Base(path), ocrErr)
	} else {
		fmt.Printf("  OCR returned too little text for %s, falling back to vision model\n", filepath.Base(path))
	}

	imgData, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading image file: %w", err)
	}

	visionModel := cfg.VisionModel
	if visionModel == "" {
		visionModel = "llava"
	}

	opts := []ollama.Option{ollama.WithModel(visionModel)}
	if cfg.OllamaURL != "" {
		opts = append(opts, ollama.WithServerURL(cfg.OllamaURL))
	}

	llm, err := ollama.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("creating vision model client: %w", err)
	}

	content := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryPart(fmt.Sprintf("image/%s", imageMediaType(ext)), imgData),
				llms.TextPart("Describe this image in detail. Extract all text, data, diagrams, and visual information. Be thorough and specific."),
			},
		},
	}

	resp, err := llm.GenerateContent(ctx, content,
		llms.WithMaxTokens(1024),
		llms.WithRepetitionPenalty(1.1),
	)
	if err != nil {
		return nil, fmt.Errorf("describing image with vision model: %w", err)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Content == "" {
		return nil, fmt.Errorf("vision model returned empty description for %s", path)
	}

	description := resp.Choices[0].Content
	fmt.Printf("  Image described via vision: %s (%d chars)\n", filepath.Base(path), len(description))

	return []schema.Document{{
		PageContent: description,
		Metadata: map[string]any{
			"source":    path,
			"file_type": ext,
			"type":      "image_description",
		},
	}}, nil
}

func imageMediaType(ext string) string {
	switch ext {
	case ".png":
		return "png"
	case ".gif":
		return "gif"
	case ".webp":
		return "webp"
	default:
		return "jpeg"
	}
}

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
	resp.Body.Close()

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
