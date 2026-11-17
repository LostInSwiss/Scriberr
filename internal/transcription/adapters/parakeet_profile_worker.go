package adapters

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// Versioned source is embedded in every build and refreshed by normal setup/update.
// This never modifies the original standard or buffered worker scripts.
func (p *ParakeetAdapter) installProfileWorker() error {
	target := filepath.Join(p.envPath, "profile-worker-v1")
	if err := os.MkdirAll(target, 0755); err != nil {
		return err
	}
	entries, err := parakeetWorkerFiles.ReadDir("parakeet_worker")
	if err != nil {
		return err
	}
	for _, e := range entries {
		content, err := parakeetWorkerFiles.ReadFile("parakeet_worker/" + e.Name())
		if err != nil {
			return err
		}
		path := filepath.Join(target, e.Name())
		current, _ := os.ReadFile(path)
		if bytes.Equal(current, content) {
			continue
		}
		temp, err := os.CreateTemp(target, "worker-*.tmp")
		if err != nil {
			return err
		}
		name := temp.Name()
		if _, err = temp.Write(content); err != nil {
			temp.Close()
			os.Remove(name)
			return err
		}
		if err = temp.Close(); err != nil {
			os.Remove(name)
			return err
		}
		if err = os.Rename(name, path); err != nil {
			os.Remove(name)
			return err
		}
	}
	return nil
}

func parakeetChunkSettings(params map[string]interface{}) (float64, bool, error) {
	maximum := 30.0
	if value, ok := params["parakeet_max_chunk_seconds"]; ok {
		switch v := value.(type) {
		case float64:
			maximum = v
		case int:
			maximum = float64(v)
		default:
			return 0, false, fmt.Errorf("Parakeet maximum chunk length must be numeric")
		}
	}
	if math.IsNaN(maximum) || math.IsInf(maximum, 0) || maximum < 5 || maximum > 120 {
		return 0, false, fmt.Errorf("Parakeet maximum chunk length must be finite and between 5 and 120 seconds")
	}
	aware := false
	if v, ok := params["parakeet_speech_aware"]; ok {
		var valid bool
		aware, valid = v.(bool)
		if !valid {
			return 0, false, fmt.Errorf("Parakeet speech-aware chunking must be boolean")
		}
	}
	return maximum, aware, nil
}

func (p *ParakeetAdapter) ValidateParameters(params map[string]interface{}) error {
	if _, _, err := parakeetChunkSettings(params); err != nil {
		return err
	}
	return p.BaseAdapter.ValidateParameters(params)
}
