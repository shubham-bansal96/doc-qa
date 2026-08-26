//go:build cgo

package ingest

import (
	"fmt"

	"github.com/otiai10/gosseract/v2"
)

// ocrImage extracts text from an image using Tesseract OCR.
func ocrImage(path string) (string, error) {
	client := gosseract.NewClient()
	defer client.Close()

	client.SetVariable("debug_file", "/dev/null") //nolint:errcheck

	if err := client.SetImage(path); err != nil {
		return "", fmt.Errorf("setting image: %w", err)
	}

	client.SetPageSegMode(gosseract.PSM_AUTO)

	text, err := client.Text()
	if err != nil {
		return "", fmt.Errorf("extracting text: %w", err)
	}

	fmt.Println("text extracted from image successfully")
	return text, nil
}
