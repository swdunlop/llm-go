// Package llm describes a high level interface to large language models suitable for basic prediction tasks.
package llm

import (
	"context"
	"fmt"
)

// Register will register a named LLM implementation.
func Register(name string, fn func(map[string]any) (Interface, error), settings ...Option) {
	_, dup := implementations[name]
	if dup {
		panic(fmt.Errorf(`%w, %q`, errDuplicateImplementation{}, name))
	}
	implementations[name] = implementation{fn, settings}
}

// Settings returns a list of settings that can be used to configure an LLM implementation.
func Settings(name string) []Option {
	imp, ok := implementations[name]
	if !ok {
		return nil
	}
	return imp.settings
}

// A Option describes a setting that can be used to configure an LLM implementation.
type Option struct {
	// Name is the name of this setting.  This is used as the key in the settings map passed to the LLM implementation.
	Name string `json:"name"` // The name of this setting.

	// Value is the value of this setting.  This is either the default value or the current value, depending on the
	// context.
	Value any `json:"value"`

	// Use describes the purpose of this setting.
	Use string `json:"use"`

	// Init identifies options that are only applicable when creating a new  instance and not when using its methods.
	Init bool `json:"init,omitempty"`
}

// errDuplicateImplementation is returned when an implementation is registered with a name that is already in use.
type errDuplicateImplementation struct{}

// Error implements the error interface by returning a static string, "duplicate implementation"
func (errDuplicateImplementation) Error() string { return "duplicate implementation" }

// New uses the named implementation to create a new LLM instance.
func New(implementation string, settings map[string]any) (Interface, error) {
	imp, ok := implementations[implementation]
	if !ok {
		return nil, fmt.Errorf(`%w, %q`, errUnknownImplementation{}, implementation)
	}
	return imp.fn(settings)
}

// implementations maps implementation names to their factory functions.
var implementations = map[string]implementation{}

type implementation struct {
	fn       func(map[string]any) (Interface, error)
	settings []Option
}

// errUnknownImplementation is returned when an unknown implementation is requested.
type errUnknownImplementation struct{}

// Error implements the error interface by returning a static string, "unknown implementation"
func (errUnknownImplementation) Error() string { return "unknown implementation" }

// Interface describes the common interface that large language models supported by this package provide.
type Interface interface {
	Release() // Closes the model and releases any associated resources.

	// Predict calls the provided function with the language model's predicted continuation of the provided input
	// string.  Prediction will stop if the function returns an error, and will eventually stop after the provided
	// context is cancelled.
	Predict(
		ctx context.Context, settings map[string]any, content []string, fn func(Prediction) error,
	) (string, error)
}

// A Prediction provides a partial prediction of the input continuation from a Predictor.
type Prediction interface {
	// String will return the predicted continuation as a string.
	String() string
}
