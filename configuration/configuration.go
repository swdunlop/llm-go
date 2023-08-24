package configuration

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
)

// Unmarshal uses the provided configuration to unmarshal the exported fields of v, which must be a pointer to a struct.
// The names of the fields are used to look up configuration values, but are overridden by the following struct tags,
// in order of precedence: "llm", "yaml", "json".
//
// Like with Get, unconfigured fields are left unchanged and the supported field types are string, []string, bool, int
// and floats.
func Unmarshal(v interface{}, cf Interface) error {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return fmt.Errorf(`unmarshal can only be used with struct pointers, got %T`, v)
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf(`unmarshal can only be used with struct pointers, got %T`, v)
	}
	rt := rv.Type()
	nf := rt.NumField()
	for i := 0; i < nf; i++ {
		ft := rt.Field(i)
		if ft.PkgPath != `` {
			continue
		}
		name := ft.Name
		if tag := ft.Tag.Get(`llm`); tag != `` {
			name = strings.SplitN(tag, `,`, 2)[0]
		} else if tag := ft.Tag.Get(`yaml`); tag != `` {
			name = strings.SplitN(tag, `,`, 2)[0]
		} else if tag := ft.Tag.Get(`json`); tag != `` {
			name = strings.SplitN(tag, `,`, 2)[0]
		}
		if name == `` || name == `-` {
			continue
		}
		if err := Get(rv.Field(i).Addr().Interface(), cf, name); err != nil {
			return err
		}
	}
	return nil
}

// Get resolves ref from a named configuration item.  Ref must be a pointer to a string, a slice of strings, a boolean,
// an integer or a float.  If the item is nil, then the ref is left unchanged.
func Get(ref any, cf Interface, name string) error {
	values := cf.GetConfiguration(name)
	if values == nil {
		return nil
	}

	if ref, ok := ref.(*[]string); ok {
		*ref = values
		return nil
	}
	if len(values) == 0 {
		return nil
	}
	if len(values) > 1 {
		return fmt.Errorf(`only one value allowed for %q, got %d`, name, len(values))
	}
	value := values[0]

	switch ref := ref.(type) {
	case *string:
		*ref = value
	case *bool:
		switch value {
		case `true`, `TRUE`, `YES`, `ON`, `1`:
			*ref = true
		case `false`, `FALSE`, `NO`, `OFF`, `0`:
			*ref = false
		default:
			return fmt.Errorf(`invalid boolean value %q for %q`, value, name)
		}
	case *int:
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf(`%w for %q`, err, name)
		}
		*ref = n
	case *float64:
		n, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf(`%w for %q`, err, name)
		}
		*ref = n
	case *float32:
		n, err := strconv.ParseFloat(value, 32)
		if err != nil {
			return fmt.Errorf(`%w for %q`, err, name)
		}
		*ref = float32(n)
	default:
		return fmt.Errorf(`unsupported type %T for %q`, ref, name)
	}
	return nil
}

// An Overlay combines multiple configurations, possibly of different types, into
// a single configuration.  The first item offering a named configuration item "wins"
// so it is best to put them in priority order, such as "Overlay(req.Configuration, Environment(`llm_`), svc.Configuration)".
type Overlay []Interface

func (cf Overlay) GetConfiguration(name string) []string {
	for _, it := range cf {
		if it == nil {
			continue
		}
		ret := it.GetConfiguration(name)
		if ret != nil {
			return ret
		}
	}
	return nil
}

// Interface describes a generic interface for providing a configuration for
// named options.
type Interface interface {
	// Get returns the set of configured values for a given name.  This typically
	// returns either nil or a slice with a single string value.  Nil indicates
	// that there was no configuration, and therefore a default can be used.
	GetConfiguration(name string) []string
}

// Map uses a map of strings to provide a configuration.  This is
// useful for inclusion in a JSON or YAML structure, since it knows how to marshal
// and unmarshal itself.
type Map map[string][]string

// GetConfiguration returns the value for the given name, or nil if there is no value.
func (cf Map) GetConfiguration(name string) []string { return cf[name] }

// UnmarshalJSON satisfies json.Unmarshaler using a map of values, where each value
// may be a string, a number, true, false, null, or an array of strings.
func (cf *Map) UnmarshalJSON(p []byte) error { panic(`TODO`) }

// MarshalJSON satisfies json.Marshaler by encoding a map of values, where each
// value is either a string, true, false, null, or an array of strings.  This is
// not perfectly symmetric with UnmarshalJSON, which also accepts numbers, which this
// marshaller will not produce.
func (cf Map) MarshalJSON() ([]byte, error) { panic(`TODO`) }

// Environment provides a configuration that gets values from the OS environment,
// adding the provided prefix to each lookup.  This is the most common source of
// configuration.  If the prefix is uppercase, then names will be converted to
// uppercase before lookup.
func Environment(prefix string) Interface {
	if prefix == `` {
		return configFn(func(name string) []string {
			value, ok := os.LookupEnv(name)
			if !ok {
				return nil
			}
			return []string{value}
		})
	} else if strings.ToUpper(prefix) == prefix {
		return configFn(func(name string) []string {
			value, ok := os.LookupEnv(prefix + strings.ToUpper(name))
			if !ok {
				return nil
			}
			return []string{value}
		})
	} else {
		return configFn(func(name string) []string {
			value, ok := os.LookupEnv(prefix + name)
			if !ok {
				return nil
			}
			return []string{value}
		})
	}
}

type configFn func(name string) []string

func (fn configFn) GetConfiguration(name string) []string { return fn(name) }
