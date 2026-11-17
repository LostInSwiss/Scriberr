package models

import (
	"encoding/json"
	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"math"
	"path/filepath"
	"testing"
)

func TestParakeetLegacyAndValidation(t *testing.T) {
	var p WhisperXParams
	if err := json.Unmarshal([]byte(`{"model_family":"nvidia_parakeet"}`), &p); err != nil {
		t.Fatal(err)
	}
	if p.ParakeetSpeechAware || p.EffectiveParakeetMaxChunkSeconds() != 30 {
		t.Fatal("legacy mode changed")
	}
	for _, n := range []float64{0, 4.9, 120.1, math.Inf(1), math.NaN()} {
		p.ParakeetMaxChunkSeconds = &n
		if p.ValidateParakeetChunking() == nil {
			t.Fatalf("accepted %v", n)
		}
	}
	for _, n := range []float64{5, 12.75, 30, 120} {
		p.ParakeetMaxChunkSeconds = &n
		if err := p.ValidateParakeetChunking(); err != nil {
			t.Fatal(err)
		}
	}
	p.ModelFamily = "whisper"
	zero := 0.0
	p.ParakeetMaxChunkSeconds = &zero
	if err := p.ValidateParakeetChunking(); err != nil {
		t.Fatal("foreign field affected Whisper")
	}
}

func TestParakeetMigrationAndRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profiles.db")
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{})
	if err != nil {
		t.Fatal(err)
	}
	// A pre-feature table/row must migrate without enabling SA1.
	if err = db.Exec("CREATE TABLE transcription_profiles (id text primary key, name text)").Error; err != nil {
		t.Fatal(err)
	}
	db.Exec("INSERT INTO transcription_profiles(id,name) VALUES ('legacy','Old profile')")
	if err = db.AutoMigrate(&TranscriptionProfile{}); err != nil {
		t.Fatal(err)
	}
	var old TranscriptionProfile
	if err = db.First(&old, "id = ?", "legacy").Error; err != nil {
		t.Fatal(err)
	}
	if old.Parameters.ParakeetSpeechAware || old.Parameters.EffectiveParakeetMaxChunkSeconds() != 30 {
		t.Fatal("migration changed defaults")
	}
	v := 22.5
	p := TranscriptionProfile{Name: "SA1", Parameters: WhisperXParams{ModelFamily: "nvidia_parakeet", ParakeetSpeechAware: true, ParakeetMaxChunkSeconds: &v}}
	if err = db.Create(&p).Error; err != nil {
		t.Fatal(err)
	}
	sqlDB, _ := db.DB()
	sqlDB.Close()
	db, err = gorm.Open(sqlite.Open(path), &gorm.Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { sqlDB, _ := db.DB(); sqlDB.Close() }()
	var found TranscriptionProfile
	db.First(&found, "id = ?", p.ID)
	if !found.Parameters.ParakeetSpeechAware || found.Parameters.EffectiveParakeetMaxChunkSeconds() != v {
		t.Fatal("restart lost profile")
	}
	found.Parameters.ParakeetSpeechAware = false
	db.Save(&found)
	db.First(&found, "id = ?", p.ID)
	if found.Parameters.ParakeetSpeechAware {
		t.Fatal("edit could not disable SA1")
	}
}
