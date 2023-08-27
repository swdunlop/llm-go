// Package llm describes a high level interface to large language models suitable for basic prediction tasks.
package llm

import (
	"context"
	"fmt"

	"github.com/swdunlop/llm-go/configuration"
)

// RegisterPredictor will register a named LLM implementation that supports prediction.
func RegisterPredictor(name string, fn func(configuration.Interface) (Predictor, error)) {
	Register(name, func(configuration configuration.Interface) (Interface, error) {
		return fn(configuration)
	})
}

// Register will register a named LLM implementation.
func Register(name string, fn func(configuration.Interface) (Interface, error)) {
	_, dup := implementations[name]
	if dup {
		panic(fmt.Errorf(`%w, %q`, errDuplicateImplementation{}, name))
	}
	implementations[name] = fn
}

// errDuplicateImplementation is returned when an implementation is registered with a name that is already in use.
type errDuplicateImplementation struct{}

// Error implements the error interface by returning a static string, "duplicate implementation"
func (errDuplicateImplementation) Error() string { return "duplicate implementation" }

// New uses the named implementation to create a new LLM instance.
func New(implementation string, configuration configuration.Interface) (Interface, error) {
	fn, ok := implementations[implementation]
	if !ok {
		return nil, fmt.Errorf(`%w, %q`, errUnknownImplementation{}, implementation)
	}
	return fn(configuration)
}

// implementations maps implementation names to their factory functions.
var implementations = map[string]func(configuration.Interface) (Interface, error){}

// errUnknownImplementation is returned when an unknown implementation is requested.
type errUnknownImplementation struct{}

// Error implements the error interface by returning a static string, "unknown implementation"
func (errUnknownImplementation) Error() string { return "unknown implementation" }

// Interface describes the common interface that large language models supported by this package provide.
type Interface interface {
	Release() // Closes the model and releases any associated resources.
}

// A Predictor is an Interface expanded with support for predicting a continuation of the provided input text.
type Predictor interface {
	Interface

	// Predict calls the provided function with the language model's predicted continuation of the provided input
	// string.  Prediction will stop if the function returns an error, and will eventually stop after the provided
	// context is cancelled.
	Predict(context.Context, []string, func(Prediction) error) (string, error)
}

// A Prediction provides a partial prediction of the input continuation from a Predictor.
type Prediction interface {
	// String will return the predicted continuation as a string.
	String() string
}
