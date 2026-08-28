.PHONY: build run-ingest run-query run-search setup clean help

BINARY_NAME=doc-qa

## help: Show this help message
help:
	@echo "doc-qa - Document QA CLI"
	@echo ""
	@echo "Usage:"
	@echo "  make build          - Build the binary"
	@echo "  make setup          - Start prerequisite services (Ollama + Qdrant)"
	@echo "  make pull-models    - Pull required Ollama models"
	@echo "  make ingest-sample  - Ingest the sample document"
	@echo "  make run-query      - Start interactive query mode"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make test           - Run tests"
	@echo ""

## build: Compile the application binary
build:
	go build -o $(BINARY_NAME) .

## setup: Start Qdrant via Docker and Ollama
setup:
	@echo "Starting Qdrant..."
	docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant || true
	@echo "Qdrant running at http://localhost:6333"
	@echo ""
	@echo "Make sure Ollama is running: ollama serve"

## pull-models: Download required Ollama models
pull-models:
	ollama pull nomic-embed-text
	ollama pull llava

## ingest-sample: Ingest the sample document from docs/
ingest-sample:
	go run . ingest docs/sample.txt

## run-query: Start interactive RAG query session
run-query:
	go run . query

## run-search: Run a similarity search (usage: make run-search Q="your query")
run-search:
	go run . search "$(Q)"

## test: Run all tests
test:
	go test ./...

## clean: Remove build artifacts
clean:
	rm -f $(BINARY_NAME)