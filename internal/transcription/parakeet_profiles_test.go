package transcription

import (
	"scriberr/internal/models"
	"testing"
)

func TestParakeetOptionsStayOutOfWhisper(t *testing.T) {
	u := &UnifiedTranscriptionService{}
	v := 17.5
	params := models.WhisperXParams{ModelFamily: FamilyNvidiaParakeet, ParakeetSpeechAware: true, ParakeetMaxChunkSeconds: &v, Device: "cuda", Threads: 6}
	p := u.convertToParakeetParams(params)
	if p["parakeet_speech_aware"] != true || p["parakeet_max_chunk_seconds"] != v || p["threads"] != 6 {
		t.Fatal(p)
	}
	params.ModelFamily = "whisper"
	for key := range u.convertToWhisperXParams(params) {
		if key == "parakeet_speech_aware" || key == "parakeet_max_chunk_seconds" {
			t.Fatal("Parakeet option leaked into Whisper")
		}
	}
}
