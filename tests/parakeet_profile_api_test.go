package tests

import (
	"encoding/json"
	"fmt"
	"scriberr/internal/models"
)

func (suite *APIHandlerTestSuite) TestParakeetProfilePersistenceAndValidation() {
	for _, value := range []interface{}{0, 121, "thirty"} {
		w := suite.makeAuthenticatedRequest("POST", "/api/v1/profiles/", map[string]interface{}{"name": "invalid", "parameters": map[string]interface{}{"model_family": "nvidia_parakeet", "parakeet_max_chunk_seconds": value}}, true)
		suite.Equal(400, w.Code)
	}
	body := map[string]interface{}{"name": "Speech aware", "parameters": map[string]interface{}{"model_family": "nvidia_parakeet", "parakeet_speech_aware": true, "parakeet_max_chunk_seconds": 22.5, "device": "cuda", "threads": 6}}
	w := suite.makeAuthenticatedRequest("POST", "/api/v1/profiles/", body, true)
	suite.Equal(200, w.Code)
	var profile models.TranscriptionProfile
	suite.NoError(json.Unmarshal(w.Body.Bytes(), &profile))
	suite.True(profile.Parameters.ParakeetSpeechAware)
	w = suite.makeAuthenticatedRequest("GET", fmt.Sprintf("/api/v1/profiles/%s", profile.ID), nil, true)
	suite.Equal(200, w.Code)
	suite.NoError(json.Unmarshal(w.Body.Bytes(), &profile))
	suite.Equal(22.5, profile.Parameters.EffectiveParakeetMaxChunkSeconds())
	profile.Parameters.ParakeetSpeechAware = false
	w = suite.makeAuthenticatedRequest("PUT", fmt.Sprintf("/api/v1/profiles/%s", profile.ID), profile, true)
	suite.Equal(200, w.Code)
	var updated models.TranscriptionProfile
	suite.NoError(json.Unmarshal(w.Body.Bytes(), &updated))
	suite.False(updated.Parameters.ParakeetSpeechAware)
}
