package kmp

// Overlap returns the size and offset of the largest overlap of pattern in space in O(n) time.
// Returns (0, 0) if no overlap is found.
func Overlap[T comparable](pattern, space []T) (max, pos int) {
	patternSz, spaceSz := len(pattern), len(space)
	if patternSz == 0 || spaceSz == 0 {
		return
	}

	table := buildKMP(pattern)
	// search for the longest prefix of pattern in space.
	for i, j := 0, 0; i < spaceSz; i++ {
		for j > 0 && space[i] != pattern[j] {
			j = table[j-1]
		}
		if space[i] == pattern[j] {
			j++
		}
		if j > max {
			max, pos = j, i-j+1
		}
		if j == patternSz {
			return
		}
	}
	return
}

func buildKMP[T comparable](pattern []T) []int {
	n := len(pattern)
	table := make([]int, n)
	for i := 1; i < n; i++ {
		j := table[i-1]
		for j > 0 && pattern[i] != pattern[j] {
			j = table[j-1]
		}
		if pattern[i] == pattern[j] {
			j++
		}
		table[i] = j
	}
	return table
}

/* This is the O(n^2) version.
for i := 0; i < len(space); i++ {
	for j := 0; j < len(pattern); j++ {
		if i+j >= len(space) {
			return
		}
		if space[i+j] != pattern[j] {
			break
		}
		if j+1 > sz {
			sz, ofs = j+1, i
		}
	}
}
return
*/
