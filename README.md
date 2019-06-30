# shmem [![Crates.io](https://img.shields.io/crates/v/shmem.svg)](https://crates.io/crates/shmem) [![Documentation](https://docs.rs/shmem/badge.svg)](https://docs.rs/shmem)

The shmem crate provides a `Shared<T>` type that handles safely storing a `T` in
shared memory. `T` is for some methods bound to be
[`ProcSync`](https://docs.rs/interprocess-traits/latest/interprocess_traits/trait.ProcSync.html)
in order to be shared between multiple processes.

Read the [documentation] for in-depth information.

[documentation]: https://docs.rs/shmem
