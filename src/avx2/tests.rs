#[cfg(test)]
mod tests {
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
}
