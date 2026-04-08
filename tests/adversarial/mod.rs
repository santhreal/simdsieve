#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::panic,
    clippy::manual_let_else
)]
mod catastrophic_alignment;
mod dense_noise;
mod fuzzer;
mod oom_survival;
mod os_page_fault;
mod prefix_aliasing;
