package llama

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

type modelFamily string

type modelType uint32

const (
	modelType3B  modelType = 26
	modelType7B  modelType = 32
	modelType13B modelType = 40
	modelType30B modelType = 60
	modelType65B modelType = 80
)

func (mt modelType) String() string {
	switch mt {
	case modelType3B:
		return "3B"
	case modelType7B:
		return "7B"
	case modelType13B:
		return "13B"
	case modelType30B:
		return "30B"
	case modelType65B:
		return "65B"
	default:
		return "Unknown"
	}
}

type fileType interface {
	String() string
}

type ggml struct {
	magic uint32
	container
	model
}

type model interface {
	ModelFamily() modelFamily
	ModelType() modelType
	FileType() fileType
}

type container interface {
	Name() string
	Decode(io.Reader) error
}

type containerGGML struct {
}

func (c *containerGGML) Name() string {
	return "ggml"
}

func (c *containerGGML) Decode(r io.Reader) error {
	return nil
}

type containerGGMF struct {
	version uint32
}

func (c *containerGGMF) Name() string {
	return "ggmf"
}

func (c *containerGGMF) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

type containerGGJT struct {
	version uint32
}

func (c *containerGGJT) Name() string {
	return "ggjt"
}

func (c *containerGGJT) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1, 2, 3:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

type containerLORA struct {
	version uint32
}

func (c *containerLORA) Name() string {
	return "ggla"
}

func (c *containerLORA) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

const (
	// / Magic constant for `ggml` files (unversioned).
	fileMagicGGML = 0x67676d6c
	// / Magic constant for `ggml` files (versioned, ggmf).
	fileMagicGGMF = 0x67676d66
	// / Magic constant for `ggml` files (versioned, ggjt).
	fileMagicGGJT = 0x67676a74
	// / Magic constant for `ggla` files (LoRA adapter).
	fileMagicGGLA = 0x67676C61
)

func decodeGGML(r io.ReadSeeker, hint modelFamily) (*ggml, error) {
	var file ggml
	binary.Read(r, binary.LittleEndian, &file.magic)

	switch file.magic {
	case fileMagicGGML:
		file.container = &containerGGML{}
	case fileMagicGGMF:
		file.container = &containerGGMF{}
	case fileMagicGGJT:
		file.container = &containerGGJT{}
	case fileMagicGGLA:
		file.container = &containerLORA{}
	default:
		return nil, errors.New("invalid file magic")
	}

	if err := file.Decode(r); err != nil {
		return nil, err
	}

	// different model types may have different layouts for hyperparameters
	switch hint {
	case modelFamilyLlama:
		var llama llamaModel
		binary.Read(r, binary.LittleEndian, &llama.hyperparameters)
		file.model = &llama
		// TODO: sanity check hyperparameters
	default:
		return nil, fmt.Errorf("unsupported model type: %s", hint)
	}

	// final model type
	return &file, nil
}
