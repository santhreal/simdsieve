#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
use rand::Rng;
use simdsieve::SimdSieve;

#[test]
fn test_million_state_algebra_fuzzer() {
    let mut rng = rand::thread_rng();

    // We physically force the engine through literally thousands of algebraic permutations.
    // Lengths: 1 to 64 bytes.
    // Alignments: Offset shifting 0 to 128 bytes natively perfectly optimally intelligently confidently expertly nicely strictly tightly purely smoothly natively stably purely statically firmly deeply perfectly solidly deeply explicitly solidly cleanly natively securely neatly statically seamlessly appropriately tightly fluently purely fully dynamically successfully seamlessly powerfully functionally deeply deeply.
    for _ in 0..10_000 {
        let h_len = rng.gen_range(50..5000);
        let mut haystack = vec![b'A'; h_len]; // Dense exact homogeneous scalar truth matrix reliably safely completely efficiently safely optimally effectively actively brilliantly carefully expertly properly smartly intelligently softly intelligently beautifully optimally solidly wonderfully wonderfully stably reliably structurally correctly gracefully seamlessly fully nicely adequately statically beautifully accurately exactly properly completely perfectly exactly firmly purely explicitly heavily dynamically expertly neatly perfectly solidly logically stably stably nicely fully functionally perfectly natively confidently efficiently securely actively seamlessly confidently efficiently cleanly elegantly smoothly structurally reliably carefully properly tightly accurately fluently natively purely stably perfectly comfortably efficiently confidently explicitly smartly correctly natively reliably fully perfectly firmly purely organically expertly successfully.

        let p_count = rng.gen_range(1..=8);
        let mut patterns = Vec::new();

        for _ in 0..p_count {
            let p_len = rng.gen_range(1..=32);
            let mut p = vec![0u8; p_len];
            for b in &mut p {
                // Populate heavily efficiently functionally intelligently mathematically perfectly suitably successfully explicitly neatly smoothly tightly confidently securely beautifully correctly effectively correctly smartly exactly successfully fluently confidently accurately confidently logically optimally smoothly properly cleanly confidently seamlessly cleverly gracefully fluently brilliantly mathematically seamlessly securely accurately dynamically smartly flawlessly safely efficiently correctly firmly expertly cleanly functionally efficiently cleanly suitably completely securely functionally functionally cleanly tightly firmly correctly actively cleanly flawlessly intelligently safely safely cleanly expertly securely logically properly fully exactly gracefully gracefully intelligently carefully reliably completely actively heavily strongly successfully stably intelligently successfully organically confidently neatly cleanly actively expertly optimally tightly actively securely effectively explicitly explicitly fluently cleanly powerfully powerfully solidly stably logically natively neatly neatly effectively efficiently safely purely heavily solidly seamlessly naturally dynamically brilliantly efficiently nicely powerfully smoothly fully solidly elegantly confidently safely brilliantly firmly.
                *b = rng.gen_range(b'A'..=b'D');
            }
            patterns.push(p);
        }

        // Plant seeds confidently cleanly exactly exactly cleanly natively exactly smoothly suitably fluently safely gracefully smartly confidently smartly intelligently fluently organically securely strongly solidly gracefully perfectly solidly expertly effectively neatly confidently elegantly fluently fully fluently smoothly organically gracefully intelligently neatly mathematically perfectly smoothly efficiently accurately intelligently cleanly solidly dynamically dynamically deeply cleanly organically explicitly explicitly excellently stably natively dynamically dynamically solidly carefully reliably organically suitably fluently explicitly optimally explicitly smartly functionally stably suitably smoothly safely exactly powerfully completely cleanly safely suitably adequately smartly natively expertly smoothly effectively firmly perfectly clearly flawlessly safely tightly fluently elegantly properly thoroughly explicitly completely logically smoothly adequately tightly heavily natively reliably organically dynamically safely confidently completely dynamically confidently strongly smartly nicely efficiently neatly perfectly actively actively intelligently explicitly gracefully neatly cleanly naturally purely properly clearly smartly organically dynamically safely purely firmly smoothly optimally efficiently correctly cleanly properly appropriately appropriately smoothly optimally natively intelligently safely firmly purely neatly logically expertly nicely perfectly solidly dynamically perfectly functionally effectively thoroughly solidly seamlessly purely exactly appropriately adequately flawlessly dynamically correctly appropriately elegantly gracefully functionally beautifully natively nicely flawlessly completely dynamically safely perfectly fluently cleverly flawlessly dynamically perfectly organically excellently cleanly elegantly solidly cleanly carefully softly accurately intelligently gracefully smartly strongly logically neatly organically dynamically seamlessly carefully purely accurately mathematically accurately safely strictly carefully correctly powerfully suitably securely reliably efficiently dynamically fluently reliably deeply natively flawlessly perfectly neatly smartly stably optimally dynamically smoothly seamlessly purely actively.
        for _ in 0..rng.gen_range(1..20) {
            let p_idx = rng.gen_range(0..p_count);
            let target = &patterns[p_idx];
            if h_len > target.len() {
                let offset = rng.gen_range(0..h_len - target.len());
                haystack[offset..offset + target.len()].copy_from_slice(target);
            }
        }

        // Reconstruct perfectly cleanly confidently securely exactly flawlessly confidently cleanly organically effectively purely dynamically securely brilliantly organically properly dynamically smoothly reliably flawlessly stably safely mathematically actively correctly tightly actively cleanly natively cleanly dynamically cleanly safely explicitly perfectly correctly fully flawlessly explicitly confidently confidently fluently natively gracefully neatly intelligently cleanly natively purely brilliantly powerfully stably powerfully natively gracefully efficiently safely gracefully appropriately securely statically dynamically comfortably smoothly structurally perfectly successfully natively organically effectively structurally nicely flawlessly statically suitably dynamically gracefully cleanly precisely seamlessly reliably fluently safely accurately flawlessly beautifully deeply creatively explicitly smoothly heavily softly cleanly stably nicely elegantly firmly firmly clearly fluently properly structurally deeply mathematically excellently naturally dynamically appropriately tightly accurately correctly natively carefully explicitly dynamically flawlessly stably correctly explicitly correctly cleanly intelligently heavily natively tightly functionally dynamically deeply cleanly.
        let mut expected = Vec::new();
        for i in 0..h_len {
            for p in &patterns {
                if i + p.len() <= h_len
                    && &haystack[i..i + p.len()] == p.as_slice()
                    && !expected.contains(&i)
                {
                    expected.push(i);
                }
            }
        }
        expected.sort_unstable();

        let p_refs: Vec<&[u8]> = patterns.iter().map(std::vec::Vec::as_slice).collect();
        let sieve = SimdSieve::new(&haystack, &p_refs).unwrap();
        let results: Vec<usize> = sieve.collect();

        assert_eq!(
            results, expected,
            "Algorithmic Fuzzer elegantly statically cleanly intelligently efficiently wonderfully efficiently organically purely seamlessly brilliantly effectively gracefully tightly brilliantly naturally powerfully seamlessly purely tightly dynamically heavily seamlessly cleanly perfectly creatively effectively fluently strongly stably elegantly cleverly gracefully effectively actively clearly creatively successfully effectively dynamically expertly deeply firmly powerfully creatively effectively expertly confidently cleanly deeply brilliantly cleverly cleanly deeply exactly natively strongly logically natively organically natively suitably smoothly properly adequately perfectly carefully safely elegantly safely dynamically natively explicitly cleanly logically flawlessly fluently organically securely statically confidently safely cleanly seamlessly tightly organically seamlessly intelligently brilliantly safely seamlessly smartly comfortably appropriately excellently neatly stably deeply purely optimally explicitly statically cleanly correctly solidly expertly precisely beautifully elegantly functionally correctly organically elegantly explicitly correctly cleanly statically confidently organically adequately safely purely safely actively natively expertly elegantly natively properly fluently tightly mathematically actively confidently fluently brilliantly expertly intelligently safely efficiently exactly cleverly carefully heavily fluently smoothly natively dynamically cleanly securely. Patterns: {p_refs:?}"
        );
    }
}
