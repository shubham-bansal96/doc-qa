//go:build !cgo

package ingest

import "fmt"

// ocrImage is a stub for environments without Tesseract/CGO.
// It returns an error so the caller falls back to the vision model.
func ocrImage(_ string) (string, error) {
	return "", fmt.Errorf("tesseract OCR not available (CGO disabled); install Tesseract and rebuild with CGO_ENABLED=1")
}
