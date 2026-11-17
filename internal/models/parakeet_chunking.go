package models

import (
	"fmt"
	"math"
)

func (p WhisperXParams) EffectiveParakeetMaxChunkSeconds() float64 {
	if p.ParakeetMaxChunkSeconds == nil {
		return 30
	}
	return *p.ParakeetMaxChunkSeconds
}

// ValidateParakeetChunking deliberately ignores inapplicable fields for other engines.
func (p WhisperXParams) ValidateParakeetChunking() error {
	if p.ModelFamily != "nvidia_parakeet" {
		return nil
	}
	v := p.EffectiveParakeetMaxChunkSeconds()
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 5 || v > 120 {
		return fmt.Errorf("Parakeet maximum chunk length must be finite and between 5 and 120 seconds")
	}
	if p.Threads < 0 || p.Threads > 64 {
		return fmt.Errorf("Parakeet threads must be between 0 and 64")
	}
	if p.Device != "" && p.Device != "auto" && p.Device != "cpu" && p.Device != "cuda" {
		return fmt.Errorf("Parakeet device must be auto, cpu or cuda")
	}
	if p.DeviceIndex < 0 {
		return fmt.Errorf("Parakeet device index must be non-negative")
	}
	return nil
}
