// Package msg describes the protocol used between the NATS client and worker.
package msg

// WorkerRequest is sent to the worker on the worker subject ask it to do something on behalf of the client.  Only one
// of the pointer fields should be non-nil.
type WorkerRequest struct {
	// Job identifies the request, which will be present in responses and can be used to cancel long running requests.
	Job string `json:"job,omitempty"`

	// Predict is a request to predict the next token.
	Predict *PredictRequest `json:"predict,omitempty"`

	// Interrupt is a request to interrupt a long running request.
	Interrupt *InterruptRequest `json:"interrupt,omitempty"`
}

// InterruptRequest is sent to the worker on the worker subject to request an interruption.
type InterruptRequest struct {
	// intentionally left blank.
}

// PredictRequest is sent to the worker on the worker subject to request a prediction.
type PredictRequest struct {
	// System contains the portion of the input that should be kept regardless of token context limits.  Warning:
	// if the System text is larger than the token context limits, the model will return an error.
	System string `json:"system,omitempty"`

	// Input contains the portion of the input that should be used to predict tokens.  Input may be truncated, losing
	// lines of text from the beginning of the input to fit within the token context limits.
	Input []string `json:"input,omitempty"`

	// Stream identifies a NATS subject where incremental results should be sent as they are produced.  This subject will
	// receive a stream of PredictStream messages in JSON format.
	//
	// No more Stream messages will be sent after the PredictResponse is sent.
	Stream string `json:"stream,omitempty"`

	// Options provides additional options used by the worker to perform the prediction.  This may be omitted or nil.
	Options map[string]string `json:"options,omitempty"`
}

// WorkerResponse is sent from the worker on the worker subject to reply to a WorkerRequest.  Only one of the pointer
// fields should be non-nil.  An empty WorkerResponse is sent if the response will be streamed.
type WorkerResponse struct {
	// Job matches the job id from the WorkerRequest.
	Job string `json:"job,omitempty"`

	// Predict is a response to a PredictRequest.
	Predict *PredictResponse `json:"predict,omitempty"`

	// Interrupt is a response to an InterruptRequest.
	Interrupt *InterruptResponse `json:"interrupt,omitempty"`

	// Stream is a response to a PredictRequest that contains incremental results.
	Stream *StreamResponse `json:"stream,omitempty"`

	// Error is a response to any request that failed.
	Error *Error `json:"error,omitempty"`
}

// StreamResponse is optionally sent if the Stream subject is provided in a PredictRequest.
type StreamResponse struct {
	// Output contains the predicted text.
	Output string `json:"output,omitempty"`
}

// String implements the fmt.Stringer and llm.Predictor interface by returning the Output field.
func (msg StreamResponse) String() string { return msg.Output }

// PredictResponse is sent as a reply to PredictRequest once the prediction is complete.
type PredictResponse struct {
	// Output contains the predicted text.
	Output string `json:"output,omitempty"`
}

// String implements the fmt.Stringer and llm.Predictor interface by returning the Output field.
func (msg PredictResponse) String() string { return msg.Output }

// InterruptResponse is sent as a reply to InterruptRequest once the interruption is complete.
type InterruptResponse struct {
	// intentionally left blank.
}

// Error is used to indicate that a request failed.
type Error struct {
	Code int    `json:"code,omitempty"`
	Err  string `json:"error"`
}

// Error implements the error interface by returning the Err field, ignoring the Code field.
func (e Error) Error() string {
	return e.Err
}

// Error codes.
const (
	ErrUnknown            = iota // omitted error code, indicates an unknown error
	ErrIllegibleRequest          // request was not a valid JSON object
	ErrInvalidRequest            // request is missing required fields or has invalid values
	ErrUnsupportedCommand        // command was not found
	ErrJobNotFound               // job was not found, usually due to interrupt
	ErrShuttingDown              // server is shutting down and will not accept new jobs
	ErrBusy                      // server is busy and cannot accept new jobs at this time
	ErrPredictionFailed          // prediction failed
	ErrInterrupted               // request was interrupted
	ErrHookFailed                // a request hook failed
)
