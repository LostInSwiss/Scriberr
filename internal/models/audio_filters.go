package models

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

const maxAudioFilterLength = 2048

var audioFilterValue = regexp.MustCompile(`^(?:[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:dB)?|true|false)$`)

// Allow only options that work on one stream without external resources or timeline edits.
var audioFilterOptions = map[string][]string{
	"acompressor": {"threshold", "ratio", "attack", "release", "makeup", "knee"},
	"afftdn":      {"nr", "nf", "tn"},
	"dynaudnorm":  {"f", "g", "p", "m", "r", "t"},
	"highpass":    {"f", "frequency", "p", "poles"},
	"loudnorm":    {"I", "TP", "LRA"},
	"lowpass":     {"f", "frequency", "p", "poles"},
	"volume":      {"volume"},
}

func (p WhisperXParams) ValidateTranscriptionOptions() error {
	if err := p.ValidateParakeetChunking(); err != nil {
		return err
	}
	return p.ValidateAudioFilters()
}

func (p WhisperXParams) ValidateAudioFilters() error {
	chain := strings.TrimSpace(p.FFmpegAudioFilters)
	if chain == "" {
		return nil
	}
	if p.ModelFamily == "openai" {
		return fmt.Errorf("FFmpeg audio filters are available for local models only")
	}
	if len(chain) > maxAudioFilterLength {
		return fmt.Errorf("FFmpeg audio filters must be at most %d characters", maxAudioFilterLength)
	}
	for _, filter := range strings.Split(chain, ",") {
		name, arguments, hasArguments := strings.Cut(strings.TrimSpace(filter), "=")
		name = strings.TrimSpace(name)
		options, allowed := audioFilterOptions[name]
		if !allowed {
			return fmt.Errorf("unsupported FFmpeg audio filter %q", name)
		}
		if !hasArguments {
			continue
		}
		for _, argument := range strings.Split(arguments, ":") {
			key, value, named := strings.Cut(argument, "=")
			if !named && (name == "volume" || name == "highpass" || name == "lowpass") && !strings.Contains(arguments, ":") {
				value, key = key, options[0]
			}
			key = strings.TrimSpace(key)
			if !slices.Contains(options, key) || !audioFilterValue.MatchString(strings.TrimSpace(value)) {
				return fmt.Errorf("unsupported FFmpeg audio filter option %q for %s", key, name)
			}
		}
	}
	return nil
}
