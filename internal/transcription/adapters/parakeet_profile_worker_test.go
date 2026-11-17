package adapters

import (
	"os"
	"path/filepath"
	"scriberr/internal/transcription/interfaces"
	"strings"
	"testing"
)

func TestProfileWorkerArgumentsAndLegacy(t *testing.T) {
	t.Setenv("PARAKEET_CHUNK_THRESHOLD_SECS", "999")
	p := NewParakeetAdapter(t.TempDir())
	for _, tc := range []struct {
		params  map[string]interface{}
		maximum string
		aware   bool
	}{
		{map[string]interface{}{}, "30", false},
		{map[string]interface{}{"parakeet_speech_aware": true, "parakeet_max_chunk_seconds": 22.5, "device": "cuda", "threads": 6}, "22.5", true},
	} {
		args, err := p.buildBufferedArgs(interfaces.AudioInput{FilePath: "any-file.wav"}, tc.params, "result-dir")
		if err != nil {
			t.Fatal(err)
		}
		s := strings.Join(args, " ")
		if !strings.Contains(s, "--chunk-len "+tc.maximum) || strings.Contains(s, "999") {
			t.Fatal(args)
		}
		if strings.Contains(s, "--speech-aware") != tc.aware {
			t.Fatal(args)
		}
		if !strings.Contains(s, "profile-worker-v1") {
			t.Fatal(args)
		}
	}
	for _, bad := range []interface{}{0.0, 121.0, "30", true} {
		_, err := p.buildBufferedArgs(interfaces.AudioInput{}, map[string]interface{}{"parakeet_max_chunk_seconds": bad}, "tmp")
		if err == nil {
			t.Fatalf("accepted %v", bad)
		}
	}
}

func TestEmbeddedWorkerSetupRefresh(t *testing.T) {
	p := NewParakeetAdapter(t.TempDir())
	if err := p.installProfileWorker(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(p.envPath, "profile-worker-v1", "profile_worker.py")
	expected, _ := os.ReadFile(path)
	os.WriteFile(path, []byte("stale"), 0600)
	if err := p.installProfileWorker(); err != nil {
		t.Fatal(err)
	}
	actual, _ := os.ReadFile(path)
	if string(actual) != string(expected) {
		t.Fatal("setup did not update bundled worker")
	}
	if _, err := os.Stat(filepath.Join(p.envPath, "transcribe_buffered.py")); !os.IsNotExist(err) {
		t.Fatal("modified original worker")
	}
}
