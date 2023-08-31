package internal

import "fmt"

// Compact will use the provided encoder to encode the combination of prefix, content and suffix, producing a slice of
// tokens that has the following properties:
//
//  1. The number of tokens is less than or equal to max.  If this is not possible, nil is returned.
//  2. All of the tokens from the prefix and suffix are included.
//  3. As many tokens as possible from the content are included.
//
// This is done by dropping lines of text from the front of the content until it fits.  Some assumptions are made, here:
// that text is peppered with newlines, that newlines have a single unambiguous encoding, and that the encoder will
// produce the same encoding for the same input.  This is true for LLaMA encoders.
func Compact(max int, instructions []int, contents [][]int) []int {
	n := len(instructions)
	max -= len(instructions)
	if max < 0 {
		return nil // not enough space for prefix and suffix
	}
	contentSize := 0
	for _, content := range contents {
		contentSize += len(content)
	}
	buf := make([]int, n, max)
	copy(buf, instructions)
	for {
		if contentSize <= max {
			for _, content := range contents {
				buf = append(buf, content...)
			}
			return buf
		}
		if len(contents) == 1 {
			// This is desperate, since the start of the content is likely to be a fragmented token or the content
			// itself is fragmented.
			content := contents[0]
			return append(buf, content[contentSize-max:]...)
		}
		contentSize -= len(contents[0])
		contents = contents[1:]
	}
} // TODO: DELETE

// An Encoder knows how to encode a string into a slice of tokens.
type Encoder interface {
	Encode(string) []int
}

func encodingOf(e Encoder, ch rune) int {
	tokens := e.Encode(string(ch))
	if len(tokens) != 1 {
		panic(fmt.Errorf(`encoding of "%c" is not one token, it is %v`, ch, tokens))
	}
	return tokens[0]
}

func tokenIndex(slice []int, value int) int {
	for i, v := range slice {
		if v == value {
			return i
		}
	}
	return -1
}
