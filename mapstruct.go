package llm

import (
	"encoding/json"
	"os"
	"strings"
)

// Env constructs a configuration map from OS environment variables with the provided prefix, like "llm_"
func Env(prefix string) map[string]any {
	out := make(map[string]any)
	for _, env := range os.Environ() {
		if !strings.HasPrefix(env, prefix) {
			continue
		}
		env = env[len(prefix):]
		ix := strings.IndexByte(env, '=')
		if ix < 0 {
			continue
		}
		name, env := env[:ix], env[ix+1:]
		if env == `` {
			continue // empty strings are ignored.
		}
		var v any
		err := json.Unmarshal([]byte(env), &v)
		if err == nil {
			out[env] = v
			continue
		}
		out[name] = env
	}
	return out
}

// Map will marshal the provided structure pointer into a map.
func Map(ref any, out map[string]any) error {
	js, err := json.Marshal(ref)
	if err != nil {
		return err
	}
	mid := make(map[string]any)
	err = json.Unmarshal(js, &mid)
	if err != nil {
		return err
	}
	for k, v := range mid {
		out[k] = v
	}
	return nil
}

// Unmap will unmarshal the provided map into the provided structure pointer.
func Unmap(in map[string]any, ref any) error {
	js, err := json.Marshal(in)
	if err != nil {
		return err
	}
	os.Stderr.Write(js)
	return json.Unmarshal(js, ref)
}
