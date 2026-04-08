#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
#[cfg(unix)]
use simdsieve::SimdSieve;
#[cfg(unix)]
use std::ptr;

#[cfg(unix)]
#[test]
fn test_catastrophic_alignment_exhaustion() {
    // The OS Page Fault test placed memory strictly at the end of a page.
    // This test exhaustively checks EVERY single byte alignment offset across the
    // 64-byte AVX-512 register window against a PROT_NONE guard page.
    // We physically iterate the start pointer from offset 0 to 63 inside a mapped OS page,
    // forcing `_mm512_loadu_si512` to execute 64 radically different unaligned combinations
    // right against the kernel cliff boundary. If the math drifts by 1 byte internally, SIGSEGV.

    unsafe {
        let page_size = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let map_size = page_size * 2;
        let mem = libc::mmap(
            ptr::null_mut(),
            map_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(mem, libc::MAP_FAILED);

        let guard_page = mem.add(page_size);
        assert_eq!(libc::mprotect(guard_page, page_size, libc::PROT_NONE), 0);

        for alignment_offset in 0..64 {
            // Test size shrinks as we push it closer to the cliff
            let test_size = 120 - alignment_offset;
            let test_start = guard_page.sub(test_size);

            let slice = std::slice::from_raw_parts_mut(test_start.cast::<u8>(), test_size);
            slice.fill(b'N');

            // Place target strictly at the last 4 bytes before the guard page
            slice[test_size - 4] = b'C';
            slice[test_size - 3] = b'V';
            slice[test_size - 2] = b'E';
            slice[test_size - 1] = b'-';

            // Execute HW logic directly against the unaligned jagged cliff edge
            let sieve = SimdSieve::new(slice, &[b"CVE-"]).unwrap();
            let results: Vec<usize> = sieve.collect();

            assert_eq!(
                results,
                vec![test_size - 4],
                "Failed catastrophic alignment at offset {alignment_offset}"
            );
        }

        libc::munmap(mem, map_size);
    }
}
