package transcription

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"scriberr/internal/models"
	"scriberr/internal/transcription/interfaces"
	"scriberr/internal/transcription/registry"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type audioFilterTestAdapter struct {
	MockTranscriptionAdapter
	transcribe func(interfaces.AudioInput) (*interfaces.TranscriptResult, error)
	diarize    func(interfaces.AudioInput) (*interfaces.DiarizationResult, error)
}

func (a *audioFilterTestAdapter) Transcribe(_ context.Context, input interfaces.AudioInput, _ map[string]interface{}, _ interfaces.ProcessingContext) (*interfaces.TranscriptResult, error) {
	return a.transcribe(input)
}

func (a *audioFilterTestAdapter) Diarize(_ context.Context, input interfaces.AudioInput, _ map[string]interface{}, _ interfaces.ProcessingContext) (*interfaces.DiarizationResult, error) {
	return a.diarize(input)
}

func TestProfileAudioFilters(t *testing.T) {
	for _, name := range []string{"ffmpeg", "ffprobe"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Skipf("%s is required for the synthetic audio test", name)
		}
	}
	runFFmpeg := func(t *testing.T, args ...string) []byte {
		t.Helper()
		cmd := exec.Command("ffmpeg", append([]string{"-hide_banner", "-nostdin", "-loglevel", "error"}, args...)...)
		output, err := cmd.Output()
		if exit, ok := err.(*exec.ExitError); ok {
			t.Fatalf("ffmpeg: %v\n%s", err, exit.Stderr)
		}
		require.NoError(t, err)
		return output
	}

	root := t.TempDir()
	stereo := filepath.Join(root, "source stereo.wav")
	mono := filepath.Join(root, "source mono.wav")
	runFFmpeg(t, "-f", "lavfi", "-i", "aevalsrc=0.8*sin(2*PI*440*t)|-0.3*sin(2*PI*440*t):s=48000:d=0.512", "-c:a", "pcm_s16le", stereo)
	runFFmpeg(t, "-i", stereo, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", mono)
	const filters = "highpass=f=100,acompressor=threshold=-30dB:ratio=3"
	const convert = "aresample=16000:ochl=mono"
	pcm := func(t *testing.T, path, filters string) []byte {
		t.Helper()
		args := []string{"-i", path}
		if filters != "" {
			args = append(args, "-af", filters)
		}
		return runFFmpeg(t, append(args, "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")...)
	}
	wantPCM := pcm(t, stereo, filters+","+convert)
	require.Len(t, wantPCM, len(pcm(t, mono, "")))
	require.NotEqual(t, wantPCM, pcm(t, stereo, convert+","+filters), "fixture must distinguish the order of operations")
	require.NotEqual(t, wantPCM, pcm(t, mono, ""), "fixture must distinguish the filtered and original signals")

	reg := registry.GetRegistry()
	oldTranscriber, _ := reg.GetTranscriptionAdapter(ModelParakeet)
	oldDiarizer, _ := reg.GetDiarizationAdapter(ModelPyannote)
	adapter := &audioFilterTestAdapter{}
	registry.RegisterTranscriptionAdapter(ModelParakeet, adapter)
	registry.RegisterDiarizationAdapter(ModelPyannote, adapter)
	t.Cleanup(func() {
		if oldTranscriber != nil {
			registry.RegisterTranscriptionAdapter(ModelParakeet, oldTranscriber)
		}
		if oldDiarizer != nil {
			registry.RegisterDiarizationAdapter(ModelPyannote, oldDiarizer)
		}
	})

	seen := map[string]bool{}
	for _, tc := range []struct {
		name, source, filters string
		diarization           bool
		cancelBefore          bool
		cancelInModel         bool
		modelError            string
		wantError             string
	}{
		{name: "blank", source: mono, filters: "  "},
		{name: "filtered_stereo", source: stereo, filters: filters, diarization: true},
		{name: "filtered_compatible_wav", source: mono, filters: filters},
		{name: "ffmpeg_failure", source: stereo, filters: "highpass=f=-1", wantError: "FFmpeg audio filters"},
		{name: "cancel_before_ffmpeg", source: stereo, filters: filters, cancelBefore: true, wantError: "context canceled"},
		{name: "adapter_failure", source: stereo, filters: filters, modelError: "adapter failed", wantError: "adapter failed"},
		{name: "cancel_in_adapter", source: stereo, filters: filters, cancelInModel: true, wantError: "context canceled"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.cancelBefore {
				cancel()
			}
			original, err := os.ReadFile(tc.source)
			require.NoError(t, err)
			job := &models.TranscriptionJob{ID: tc.name, AudioPath: tc.source, Parameters: models.WhisperXParams{
				ModelFamily: FamilyNvidiaParakeet, FFmpegAudioFilters: tc.filters, Diarize: tc.diarization,
			}}
			repo := new(MockJobRepository)
			repo.On("FindWithAssociations", mock.Anything, job.ID).Return(job, nil).Once()
			repo.On("CreateExecution", mock.Anything, mock.MatchedBy(func(e *models.TranscriptionJobExecution) bool {
				return e.ActualParameters.FFmpegAudioFilters == tc.filters
			})).Return(nil).Once()
			status := models.StatusCompleted
			if tc.wantError == "" {
				repo.On("UpdateTranscript", mock.Anything, job.ID, mock.Anything).Return(nil).Once()
			} else {
				status = models.StatusFailed
			}
			repo.On("UpdateExecution", mock.Anything, mock.MatchedBy(func(e *models.TranscriptionJobExecution) bool {
				return e.Status == status
			})).Return(nil).Once()
			svc := NewUnifiedTranscriptionService(repo)
			svc.tempDirectory = root
			svc.outputDirectory = filepath.Join(root, "transcripts")
			var modelInput string
			adapter.transcribe = func(input interfaces.AudioInput) (*interfaces.TranscriptResult, error) {
				modelInput = input.FilePath
				if tc.name == "blank" {
					require.Equal(t, tc.source, input.FilePath)
				} else {
					require.Equal(t, root, filepath.Dir(input.FilePath))
					require.NotEqual(t, tc.source, input.FilePath)
					require.False(t, seen[input.FilePath], "each execution needs a unique temporary file")
					seen[input.FilePath] = true
					actual, err := svc.createAudioInput(input.FilePath)
					require.NoError(t, err)
					require.Equal(t, 16000, actual.SampleRate)
					require.Equal(t, 1, actual.Channels)
					require.Equal(t, "pcm_s16le", actual.Metadata["codec"])
					if tc.source == stereo {
						require.Equal(t, wantPCM, pcm(t, input.FilePath, ""))
					}
				}
				if tc.cancelInModel {
					cancel()
					return nil, ctx.Err()
				}
				if tc.modelError != "" {
					return nil, errors.New(tc.modelError)
				}
				return &interfaces.TranscriptResult{Text: "synthetic"}, nil
			}
			diarized := false
			adapter.diarize = func(input interfaces.AudioInput) (*interfaces.DiarizationResult, error) {
				diarized = true
				require.Equal(t, modelInput, input.FilePath)
				require.FileExists(t, input.FilePath)
				return &interfaces.DiarizationResult{}, nil
			}
			if err := svc.ProcessJob(ctx, job.ID); tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
			} else {
				require.NoError(t, err)
			}
			if tc.cancelBefore || tc.name == "ffmpeg_failure" {
				require.Empty(t, modelInput, "recognition must not run after filtering fails")
			} else {
				require.NotEmpty(t, modelInput)
			}
			require.Equal(t, tc.diarization, diarized)
			require.Equal(t, tc.source, job.AudioPath)
			after, err := os.ReadFile(tc.source)
			require.NoError(t, err)
			require.Equal(t, original, after)
			leftovers, err := filepath.Glob(filepath.Join(root, "ffmpeg-*.wav"))
			require.NoError(t, err)
			require.Empty(t, leftovers)
			repo.AssertExpectations(t)
		})
	}
}
