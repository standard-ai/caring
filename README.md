# caring [![Crates.io](https://img.shields.io/crates/v/caring.svg)](https://crates.io/crates/caring) [![Documentation](https://docs.rs/caring/badge.svg)](https://docs.rs/caring)

Sharing is caring: the caring crate provides a `Shared<T>` type that handles
safely storing a `T` in shared memory, using `mmap`. `T` is for some methods
bound to be
[`ProcSync`](https://docs.rs/interprocess-traits/latest/interprocess_traits/trait.ProcSync.html)
in order to be shared between multiple processes.

Read the [documentation] for in-depth information.

[documentation]: https://docs.rs/caring
