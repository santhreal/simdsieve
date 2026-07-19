use crate::avx2::Avx2Filter;
use crate::scalar::ScalarFilter;

#[test]
fn case_insensitive_masks_expose_pump_b_boundary_state() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let filter = unsafe { Avx2Filter::new(&[b"Z"], true) };
    let mut block = [b'x'; 65];
    block[63] = b'Z';

    let (mask_a, mask_b) = unsafe { filter.check_64byte_block(&block) };
    eprintln!("mask_a={mask_a:032b}");
    eprintln!("mask_b={mask_b:032b}");

    assert_eq!(mask_a, 0);
    assert_eq!(mask_b & (1 << 31), 1 << 31);
}

#[test]
fn avx2_64byte_block_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let patterns: &[&[u8]] = &[b"ab", b"XY", b"1"];
    let avx2 = unsafe { Avx2Filter::new(patterns, false) };
    let scalar = ScalarFilter::new(patterns, false);

    let mut block = [b'x'; 68];
    block[10] = b'a';
    block[11] = b'b';
    block[35] = b'X';
    block[36] = b'Y';
    block[63] = b'1';

    let (mask_a, mask_b) = unsafe { avx2.check_64byte_block(&block) };
    let scalar_mask = scalar.check_64byte_block(&block);
    let avx2_mask = u64::from(mask_a) | (u64::from(mask_b) << 32);

    assert_eq!(
        avx2_mask, scalar_mask,
        "AVX2 64-byte block must match scalar backend"
    );
}

#[test]
fn avx2_32byte_block_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let patterns: &[&[u8]] = &[b"te", b"ST"];
    let avx2 = unsafe { Avx2Filter::new(patterns, false) };
    let scalar = ScalarFilter::new(patterns, false);

    // Scalar check_64byte_block needs 64 + max_len - 1 bytes.
    let mut block = [b'x'; 65];
    block[5] = b't';
    block[6] = b'e';
    block[30] = b'S';
    block[31] = b'T';

    let avx2_mask = unsafe { avx2.check_32byte_block(&block) };
    let scalar_mask = scalar.check_64byte_block(&block) as u32;

    assert_eq!(
        avx2_mask, scalar_mask,
        "AVX2 32-byte block must match scalar backend low 32 bits"
    );
}

#[test]
fn avx2_case_insensitive_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let patterns: &[&[u8]] = &[b"Ab", b"z"];
    let avx2 = unsafe { Avx2Filter::new(patterns, true) };
    let scalar = ScalarFilter::new(patterns, true);

    let mut block = [b'x'; 68];
    block[15] = b'a';
    block[16] = b'B';
    block[47] = b'Z';

    let (mask_a, mask_b) = unsafe { avx2.check_64byte_block(&block) };
    let scalar_mask = scalar.check_64byte_block(&block);
    let avx2_mask = u64::from(mask_a) | (u64::from(mask_b) << 32);

    assert_eq!(
        avx2_mask, scalar_mask,
        "AVX2 case-insensitive must match scalar backend"
    );
}

/// Randomized parity over a realistic 8-pattern set with 3–4 byte prefixes and
/// shared first bytes (`AKIA`/`ASIA`, `xoxb-`/`xoxp-`, `sk-proj-`/`sq0csp-`).
/// This is the exact shape the vector-domain fold optimization targets: many
/// patterns, multi-byte prefixes, prefix collisions. Each random block is
/// checked against the scalar oracle for both the 64- and 32-byte entry points.
#[test]
fn avx2_hot_prefix_set_matches_scalar_randomized() {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let patterns: &[&[u8]] = &[
        b"ghp_",
        b"sk-proj-",
        b"AKIA",
        b"ASIA",
        b"SG.",
        b"xoxb-",
        b"xoxp-",
        b"sq0csp-",
    ];
    for &ci in &[false, true] {
        let avx2 = unsafe { Avx2Filter::new(patterns, ci) };
        let scalar = ScalarFilter::new(patterns, ci);
        let mut rng = StdRng::seed_from_u64(0x5EED_1234_5678_9ABC ^ u64::from(ci));
        // Alphabet biased toward pattern bytes so matches actually fire and the
        // OR-fold across same-first-byte patterns is exercised, not just misses.
        let alphabet = b"AKISGghpsknxob-0123_qcrojzZ aA";
        for _ in 0..20_000 {
            let mut block = [0u8; 68];
            for b in &mut block {
                *b = alphabet[rng.gen_range(0..alphabet.len())];
            }
            if rng.gen_bool(0.5) {
                let p = patterns[rng.gen_range(0..patterns.len())];
                let pos = rng.gen_range(0..=block.len() - p.len());
                block[pos..pos + p.len()].copy_from_slice(p);
            }

            let (mask_a, mask_b) = unsafe { avx2.check_64byte_block(&block) };
            let avx2_mask = u64::from(mask_a) | (u64::from(mask_b) << 32);
            let scalar_mask = scalar.check_64byte_block(&block);
            assert_eq!(
                avx2_mask, scalar_mask,
                "64-byte parity (ci={ci}) block={block:?}"
            );

            let avx2_32 = unsafe { avx2.check_32byte_block(&block) };
            let scalar_32 = scalar.check_64byte_block(&block) as u32;
            assert_eq!(
                avx2_32, scalar_32,
                "32-byte parity (ci={ci}) block={block:?}"
            );
        }
    }
}
