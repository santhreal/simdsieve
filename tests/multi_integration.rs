#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use simdsieve::MultiSieve;

const HAYSTACK_BYTES: usize = 64 * 1024;
const FIRST_PATTERN_OFFSET: usize = 1_024;
const PATTERN_SPACING: usize = 2_500;

fn build_haystack_64kb(patterns: &[Vec<u8>], filler: u8) -> Vec<u8> {
    let mut haystack = vec![filler; HAYSTACK_BYTES];

    for (index, pattern) in patterns.iter().enumerate() {
        let start = FIRST_PATTERN_OFFSET + (index * PATTERN_SPACING);
        let end = start + pattern.len();
        assert!(
            end <= haystack.len(),
            "pattern index {index} does not fit in 64KB test haystack"
        );

        haystack[start..end].copy_from_slice(pattern);
    }

    haystack
}

fn lowercase_patterns(patterns: &[Vec<u8>]) -> Vec<Vec<u8>> {
    patterns
        .iter()
        .map(|pattern| pattern.iter().map(u8::to_ascii_lowercase).collect())
        .collect()
}

fn naive_matches(haystack: &[u8], patterns: &[&[u8]], case_insensitive: bool) -> Vec<usize> {
    let mut expected = Vec::new();

    for offset in 0..haystack.len() {
        if patterns.iter().any(|pattern| {
            if pattern.is_empty() {
                return true;
            }
            if offset + pattern.len() > haystack.len() {
                return false;
            }
            if case_insensitive {
                haystack[offset..offset + pattern.len()]
                    .iter()
                    .zip(*pattern)
                    .all(|(&a, &b)| a.eq_ignore_ascii_case(&b))
            } else {
                &haystack[offset..offset + pattern.len()] == *pattern
            }
        }) {
            expected.push(offset);
        }
    }

    expected
}

#[test]
fn multi_sieve_integration_smoke_64kb_scan() {
    let patterns: Vec<Vec<u8>> = vec![
        b"alpha".to_vec(),
        b"beta_42".to_vec(),
        b"GammaRay".to_vec(),
        b"deltaForce".to_vec(),
        b"epsilon".to_vec(),
        b"zeta-omega".to_vec(),
        b"eta".to_vec(),
        b"thetaX".to_vec(),
        b"iota-long".to_vec(),
        b"kappa".to_vec(),
        b"lambda".to_vec(),
        b"mu_sigma".to_vec(),
        b"nu".to_vec(),
        b"xi-pattern".to_vec(),
        b"omicron".to_vec(),
        b"pi2".to_vec(),
        b"rho-data".to_vec(),
        b"sigma".to_vec(),
        b"tau-wave".to_vec(),
        b"upsilon".to_vec(),
        b"phi".to_vec(),
        b"chi".to_vec(),
        b"psi".to_vec(),
        b"omega".to_vec(),
    ];

    let pattern_refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();
    assert!(
        pattern_refs.len() >= 20,
        "integration pattern set must include 20+ patterns"
    );

    let haystack = build_haystack_64kb(&patterns, b'.');

    let expected = naive_matches(&haystack, &pattern_refs, false);
    let found: Vec<usize> = MultiSieve::new(&haystack, &pattern_refs)
        .expect("construct MultiSieve for integration smoke test")
        .candidates()
        .collect();

    for offset in &expected {
        assert!(
            found.binary_search(offset).is_ok(),
            "false negative at offset {offset}: pattern expected but not reported"
        );
    }

    assert!(
        expected.len() <= found.len(),
        "scan should not miss expected matches (and may include false positives by design)"
    );

    assert_eq!(
        expected, found,
        "unexpected false positives for case-sensitive 64KB MultiSieve integration scan"
    );
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn multi_sieve_case_insensitive_mode_via_normalization() {
    let mixed_case_patterns: Vec<Vec<u8>> = vec![
        b"Alpha".to_vec(),
        b"beta".to_vec(),
        b"GaMmA".to_vec(),
        b"DeLtA".to_vec(),
        b"ePsIlOn".to_vec(),
        b"ZETA".to_vec(),
        b"eta42".to_vec(),
        b"tHeta".to_vec(),
        b"iota".to_vec(),
        b"KAPPA".to_vec(),
        b"lambdaX".to_vec(),
        b"MU".to_vec(),
        b"Nu".to_vec(),
        b"XiP".to_vec(),
        b"OmIcRoN".to_vec(),
        b"PI2".to_vec(),
        b"Rho-data".to_vec(),
        b"siGmA".to_vec(),
        b"tAu".to_vec(),
        b"Upsilon".to_vec(),
        b"PHI".to_vec(),
        b"chi".to_vec(),
        b"PSI".to_vec(),
        b"oMeGa".to_vec(),
    ];

    let pattern_refs: Vec<&[u8]> = mixed_case_patterns.iter().map(Vec::as_slice).collect();
    let haystack = build_haystack_64kb(&mixed_case_patterns, b'@');
    let expected = naive_matches(&haystack, &pattern_refs, true);

    let haystack_ci = haystack
        .iter()
        .map(u8::to_ascii_lowercase)
        .collect::<Vec<u8>>();
    let lowered_patterns = lowercase_patterns(&mixed_case_patterns);
    let lowered_pattern_refs: Vec<&[u8]> = lowered_patterns.iter().map(Vec::as_slice).collect();

    let found: Vec<usize> = MultiSieve::new(&haystack_ci, &lowered_pattern_refs)
        .expect("construct MultiSieve for case-insensitive normalization smoke test")
        .candidates()
        .collect();

    for offset in &expected {
        assert!(
            found.binary_search(offset).is_ok(),
            "false negative at offset {offset}: case-insensitive match was not reported"
        );
    }

    assert_eq!(
        expected, found,
        "case-insensitive normalization path via MultiSieve should remain exact"
    );
}
