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
fn test_os_page_boundary_segfault_immunity() {
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

        let test_size = 100;
        let test_start = guard_page.sub(test_size);
        let slice = std::slice::from_raw_parts_mut(test_start.cast::<u8>(), test_size);

        slice.fill(b'N');
        slice[test_size - 4] = b'C';
        slice[test_size - 3] = b'V';
        slice[test_size - 2] = b'E';
        slice[test_size - 1] = b'-';

        let sieve = SimdSieve::new(slice, &[b"CVE-"]).unwrap();
        let results: Vec<usize> = sieve.collect();
        assert_eq!(results, vec![test_size - 4]);

        libc::munmap(mem, map_size);
    }
}
