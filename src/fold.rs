//! ASCII folding and verification helpers shared by scalar and scalarized paths.

/// Offset used to normalize lowercase ASCII bytes to uppercase.
///
/// # Example
///
/// ```
/// use simdsieve::fold::ASCII_CASE_OFFSET;
/// assert_eq!(ASCII_CASE_OFFSET, 0x20);
/// ```
pub const ASCII_CASE_OFFSET: u8 = 0x20;

/// Fold a single ASCII byte to uppercase (case-insensitive form).
///
/// Branchless conversion for `a` through `z`.
///
/// # Example
///
/// ```
/// use simdsieve::fold::fold_ascii_lowercase;
/// assert_eq!(fold_ascii_lowercase(b'a'), b'A');
/// assert_eq!(fold_ascii_lowercase(b'A'), b'A');
/// assert_eq!(fold_ascii_lowercase(b'7'), b'7');
/// ```
#[inline]
pub fn fold_ascii_lowercase(b: u8) -> u8 {
    fold_byte(b)
}

/// Fold a single ASCII byte to case-insensitive form.
///
/// Uppercase conversion is implemented branchlessly for `a` through `z`.
/// Non-ASCII bytes and already-uppercase bytes are returned unchanged.
///
/// # Example
///
/// ```
/// use simdsieve::fold::fold_byte;
/// assert_eq!(fold_byte(b'z'), b'Z');
/// assert_eq!(fold_byte(b'Z'), b'Z');
/// assert_eq!(fold_byte(0xFF), 0xFF);
/// ```
#[inline]
pub fn fold_byte(b: u8) -> u8 {
    let is_lowercase = u8::from(b.wrapping_sub(b'a') < 26);
    b.wrapping_sub(ASCII_CASE_OFFSET & is_lowercase.wrapping_neg())
}

/// Fold an entire byte slice in-place to case-insensitive form.
///
/// # Example
///
/// ```
/// use simdsieve::fold::fold_slice;
/// let mut data = *b"Hello";
/// fold_slice(&mut data);
/// assert_eq!(&data, b"HELLO");
/// ```
#[inline]
pub fn fold_slice(data: &mut [u8]) {
    for item in data.iter_mut() {
        *item = fold_byte(*item);
    }
}

/// Exact byte-by-byte comparison.
///
/// # Example
///
/// ```
/// use simdsieve::fold::verify_exact;
/// assert!(verify_exact(b"needle", b"needle"));
/// assert!(!verify_exact(b"needle", b"noodle"));
/// ```
#[inline]
pub fn verify_exact(left: &[u8], right: &[u8]) -> bool {
    left == right
}

/// Case-insensitive comparison by folding both sides with ASCII-only logic.
///
/// Returns `false` if the slices have different lengths.
///
/// # Example
///
/// ```
/// use simdsieve::fold::verify_case_insensitive;
/// assert!(verify_case_insensitive(b"Hello", b"HELLO"));
/// assert!(verify_case_insensitive(b"Rust", b"rust"));
/// assert!(!verify_case_insensitive(b"hi", b"hello"));
/// ```
#[inline]
pub fn verify_case_insensitive(pattern: &[u8], haystack: &[u8]) -> bool {
    if pattern.len() != haystack.len() {
        return false;
    }

    pattern
        .iter()
        .zip(haystack.iter())
        .all(|(&pat, &h)| fold_byte(pat) == fold_byte(h))
}

#[cfg(test)]
mod tests {
    use super::{fold_byte, fold_slice, verify_case_insensitive, verify_exact};

    #[test]
    fn fold_byte_classifies_ascii_ranges() {
        for c in b'a'..=b'z' {
            assert_eq!(fold_byte(c), c - 0x20);
        }

        for c in b'A'..=b'Z' {
            assert_eq!(fold_byte(c), c);
        }

        for c in b'0'..=b'9' {
            assert_eq!(fold_byte(c), c);
        }

        for c in *b"!-/:`{~" {
            assert_eq!(fold_byte(c), c);
        }
    }

    #[test]
    fn verify_case_insensitive_matches_folded_exact() {
        let pattern = b"AbC-9!xY";
        let haystack = b"aBc-9!Xy";
        let mut folded_pattern = pattern.to_vec();
        let mut folded_haystack = haystack.to_vec();

        fold_slice(&mut folded_pattern);
        fold_slice(&mut folded_haystack);

        assert_eq!(
            verify_case_insensitive(pattern, haystack),
            verify_exact(&folded_pattern, &folded_haystack)
        );
    }
}
