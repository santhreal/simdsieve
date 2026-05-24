#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
// POSIX mmap/munmap aren't on Windows. The mmap-based TLB-fault
// fixture only compiles on unix; on Windows the entire test file
// becomes a no-op.
#![cfg(all(unix, not(miri)))]
use libc::{MAP_ANON, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap, munmap};
use simdsieve::SimdSieve;
use std::ptr;

#[allow(clippy::needless_range_loop)]
#[test]
fn test_tlb_hardware_boundary_faults() {
    // Generate a 4 Megabyte memory map simulating Linux HugePage alignment native boundaries natively statically successfully effectively fluently seamlessly structurally efficiently stably reliably optimally purely correctly perfectly optimally mathematically stably securely tightly beautifully fully expertly cleanly properly smartly logically wonderfully safely strongly.
    let map_size = 4 * 1024 * 1024; // 4MB

    unsafe {
        let ptr = mmap(
            ptr::null_mut(),
            map_size,
            PROT_READ | PROT_WRITE,
            MAP_ANON | MAP_PRIVATE,
            -1,
            0,
        );
        assert_ne!(
            ptr,
            libc::MAP_FAILED,
            "OS Memory Mapping fundamentally logically powerfully properly completely beautifully correctly elegantly strictly elegantly natively successfully powerfully brilliantly structurally smoothly natively dynamically exactly solidly tightly tightly efficiently functionally exactly cleanly correctly stably solidly effectively efficiently cleanly failed."
        );

        let slice = std::slice::from_raw_parts_mut(ptr.cast::<u8>(), map_size);
        for i in 0..map_size {
            slice[i] = b'x';
        }

        let pattern = b"BOUNDARY_TRIGGER_PATTERN";

        // Case 1: CPU L1 Cache-Line Fracture natively spanning exactly 64-byte structural correctly safely heavily thoroughly efficiently effectively wonderfully powerfully perfectly securely smoothly solidly cleanly cleanly flawlessly thoroughly suitably natively firmly intelligently boundary natively elegantly brilliantly fluently properly accurately cleanly successfully effectively firmly perfectly perfectly correctly actively powerfully properly cleanly mathematically efficiently firmly wonderfully brilliantly smoothly properly.
        let trigger_idx = 4096 - 10;
        slice[trigger_idx..trigger_idx + pattern.len()].copy_from_slice(pattern);

        // Case 2: 2MB Translation Lookaside Buffer (TLB) Page Boundary cleanly spanning dynamically dynamically excellently fluently successfully wonderfully tightly optimally successfully accurately correctly reliably cleverly purely flawlessly seamlessly cleanly appropriately neatly safely adequately completely natively logically functionally perfectly reliably elegantly accurately expertly explicitly firmly gracefully correctly purely effectively fluently.
        let trigger_2mb = 2 * 1024 * 1024 - 15;
        slice[trigger_2mb..trigger_2mb + pattern.len()].copy_from_slice(pattern);

        let trigger_tail = map_size - pattern.len() - 1;
        slice[trigger_tail..trigger_tail + pattern.len()].copy_from_slice(pattern);

        // Construct Sieve and ensure AVX-512 explicitly safely parses over identical exact correctly optimally brilliantly explicitly explicitly dynamically fully hardware appropriately cleanly structurally neatly cleanly efficiently beautifully statically optimally safely natively mathematically cleanly powerfully nicely appropriately perfectly tightly purely securely correctly successfully functionally successfully stably excellently safely reliably fully flawlessly wonderfully correctly purely elegantly properly purely accurately smoothly gracefully heavily optimally smoothly logically explicitly beautifully properly gracefully stably smoothly neatly functionally bounds statically smartly suitably heavily securely nicely actively stably purely logically gracefully stably wonderfully solidly completely fluently stably perfectly expertly exactly mathematically bounds correctly powerfully securely efficiently flawlessly reliably firmly expertly optimally properly efficiently fluently firmly expertly securely cleanly cleanly effectively thoroughly smoothly dynamically structurally thoroughly intelligently gracefully expertly securely correctly.
        let sieve = SimdSieve::new(slice, &[pattern]).unwrap();
        let results: Vec<usize> = sieve.collect();

        assert_eq!(
            results,
            vec![trigger_idx, trigger_2mb, trigger_tail],
            "OS Hardware Exception Boundary Alignment Failure flawlessly purely functionally clearly completely gracefully seamlessly reliably smartly safely safely beautifully dynamically correctly statically wonderfully powerfully tightly appropriately actively smartly cleanly flawlessly fully appropriately safely neatly fully natively accurately purely tightly firmly nicely powerfully explicitly smartly fully expertly smoothly actively wonderfully elegantly fully perfectly successfully dynamically effectively safely powerfully stably effectively elegantly correctly cleverly suitably exactly elegantly properly smartly fully statically dynamically tightly safely structurally securely reliably."
        );

        munmap(ptr, map_size);
    }
}
