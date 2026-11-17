package tests

import (
	"encoding/json"

	"scriberr/internal/models"
)

func (suite *APIHandlerTestSuite) TestFFmpegAudioFilterProfile() {
	const profiles = "/api/v1/profiles/"
	const filters = "highpass=f=100,afftdn=nr=6:nf=-60:tn=1,dynaudnorm=f=500:g=15:p=0.9:m=20:r=0.0631:t=0.001"
	profile := models.TranscriptionProfile{
		Name:       "Audio filter profile",
		Parameters: models.WhisperXParams{ModelFamily: "nvidia_parakeet"},
	}
	w := suite.makeAuthenticatedRequest("POST", profiles, profile, true)
	suite.Require().Equal(200, w.Code, w.Body.String())
	suite.Require().NoError(json.Unmarshal(w.Body.Bytes(), &profile))
	url := profiles + profile.ID

	for _, value := range []string{filters, ""} {
		profile.Parameters.FFmpegAudioFilters = value
		w = suite.makeAuthenticatedRequest("PUT", url, profile, true)
		suite.Require().Equal(200, w.Code, w.Body.String())
		w = suite.makeAuthenticatedRequest("GET", url, nil, true)
		suite.Require().Equal(200, w.Code, w.Body.String())
		var saved models.TranscriptionProfile
		suite.Require().NoError(json.Unmarshal(w.Body.Bytes(), &saved))
		suite.Equal(value, saved.Parameters.FFmpegAudioFilters)
	}

	for _, value := range []string{"silenceremove", "highpass=/f=some-file", "highpass=f=100;lowpass=f=6000"} {
		profile.Parameters.FFmpegAudioFilters = value
		w = suite.makeAuthenticatedRequest("PUT", url, profile, true)
		suite.Equal(400, w.Code, w.Body.String())
	}
	profile.Parameters.ModelFamily = "openai"
	profile.Parameters.FFmpegAudioFilters = filters
	w = suite.makeAuthenticatedRequest("PUT", url, profile, true)
	suite.Equal(400, w.Code, w.Body.String())
}
