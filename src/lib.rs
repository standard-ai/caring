extern crate errno;
extern crate interprocess_traits;
extern crate libc;
extern crate memfd;
extern crate thiserror;

#[cfg(test)]
extern crate sendfd;

use std::{
    io, mem,
    ops::Deref,
    os::unix::io::{AsRawFd, IntoRawFd, RawFd},
    ptr,
};

use errno::errno;
use interprocess_traits::ProcSync;
use libc::c_void;
use memfd::MemfdOptions;

// TODO: Drop the contained `T` when the last `Shared<T>` handle to it is dropped.
// This will require:
//  * To add an atomic reference counter
//  * To handle properly ZST `T` used with a `size`

// TODO: Add a safer interface than AsRawFd/IntoRawFd
// This could be done by implementing a trait like `SendShared` / `RecvShared` similar to what is
// in `sendfd`, that would use the data-passing functionality of the socket to implement a protocol
// that ensures the `T` is the same on both sides.
// Note that:
//  * This will still be unsafe from a rust point of view, as the other side could respect the
//    protocol but be sending a different fd -- knowing the call is safe requires knowing the other
//    side is going to respect the protocol
//  * This will not deprecate the *RawFd series of functions, that will still be required for
//    fd-passing over interfaces that are not supported by `SendShared` / `RecvShared`

// TODO: Consider making this crate work on non-unix systems, just with the fd-passing behavior
// suppressed.
// It is not obvious that we want to actually do this, as in such cases `Shared` amounts to a less
// good `Arc`, and the user should then not be incentivized to use it.

/// The error type for errors in this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Could not create an in-memory file for shared memory
    #[error("Could not create an in-memory file for shared memory")]
    CreateMemfd(#[source] memfd::Error),

    /// Failed to set the length of the shared memory file
    #[error("Failed to set the length of the shared memory file")]
    Truncate(#[source] io::Error),

    /// Failed to retrieve the length of the shared memory file
    #[error("Failed to retrieve the length of the shared memory file")]
    GetMetadata(#[source] io::Error),

    /// File length after truncation does not match requested size
    #[error(
        "Failed to truncate the in-memory file to {expected} bytes, the file is {actual}B long"
    )]
    Length { expected: usize, actual: usize },

    /// Failed to map shared memory
    #[error("Failed to map shared memory")]
    Mmap(#[source] io::Error),

    /// Failed to duplicate the file descriptor
    #[error("Failed to duplicate the file descriptor")]
    Dup(#[source] io::Error),
}

/// A memory-mapped region pointing to an element of type T.
///
/// The memory region is owned, and will be unmapped when this is dropped. Note that the element of
/// type T will *not* be dropped.
struct MmapRegion<T> {
    /// Start of the memory-mapped region
    ptr: *mut T,

    /// Size of the memory-mapped region (usually `mem::size_of::<T>()`, but it can be greater than
    /// that for eg. `c_void`, which can be used for `!Sized` types)
    size: usize,
}

impl<T> MmapRegion<T> {
    /// Memory-maps `fd` with size `size` and returns a pointer to the mapped memory region.
    ///
    /// # Safety
    ///
    /// This assumes that `fd` points to a file of at least size `size`.
    unsafe fn new(size: usize, fd: RawFd) -> Result<MmapRegion<T>, Error> {
        // TODO: investigate the potential impact of MAP_HUGETLB | MAP_HUGE_2MB
        let ptr = libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE, // Read-write mapping
            // MAP_SHARED_VALIDATE: Share mapping with `fork`'d processes, validate that all flags are known
            // MAP_POPULATE: immediately reserve the pages, do not lazily allocate them
            libc::MAP_SHARED_VALIDATE | libc::MAP_POPULATE,
            fd,
            0, // offset
        );

        // Check the memory map succeeded
        if ptr == libc::MAP_FAILED {
            return Err(Error::Mmap(errno().into()));
        }

        Ok(MmapRegion {
            ptr: ptr as *mut T,
            size,
        })
    }
}

impl<T> Drop for MmapRegion<T> {
    fn drop(&mut self) {
        // Do not drop contents here, see type-level documentation
        unsafe {
            // This is safe thanks to the memory region being owned.
            libc::munmap(self.ptr as *mut c_void, self.size);
        }
        // For now, if `munmap` failed, we're ignoring it.
        // However it means that something happened, so we might some day want to change this, for
        // instance by logging something.
        // See also https://github.com/rust-lang/rfcs/pull/2677
    }
}

/// A wrapper for data that can be shared across processes.
///
/// The data is owned, but the element of type `T` will *not* be dropped when the `Shared<T>` is
/// dropped. Note that an element that has a meaningful `Drop` is likely not `ProcSync` anyway.
pub struct Shared<T> {
    fd: RawFd,
    region: MmapRegion<T>,
}

// These implementations are safe thanks to `Shared` giving only a reference-based access to the
// underlying `T`
unsafe impl<T: Sync> Send for Shared<T> {}
unsafe impl<T: Sync> Sync for Shared<T> {}

/// Creates an uninitialized `Shared<T>` that points to a memory region of size `size`.
///
/// Note that the anonymous file associated with this memory mapping is sealed and
/// cannot be resized. If it were not the case, `SIGBUS` could be hit if the file
/// were truncated.
///
/// # Safety
///
/// Returned `Shared` will point to uninitialized memory.
///
/// # Panics
///
/// Panics if `T` requires an alignment greater than the page size.
unsafe fn create_shared<T>(size: usize) -> Result<Shared<T>, Error> {
    // Check that alignment of T is at most one page
    let page_size = libc::sysconf(libc::_SC_PAGE_SIZE) as usize;
    let requested_align = mem::align_of::<T>();
    if requested_align > page_size {
        // Note: This is not implemented as an error, as:
        //  1. This error condition is *particularly* unlikely to ever happen, requiring both a
        //     developer with appetite for custom-align structs and a user with a somehow reduced
        //     page size.
        //  2. This makes us forwards-compatible for the day we'll decide we want to actually
        //     support that use case (by allocating lots of memory).
        panic!(
            "Page size {}B is too low for requested alignment {}",
            page_size, requested_align
        );
    }

    // Create the file
    let memfd = MemfdOptions::new()
        .allow_sealing(true)
        .close_on_exec(true)
        .create("caring")
        .map_err(Error::CreateMemfd)?;
    let file = memfd.into_file();

    // Truncate
    file.set_len(size as u64).map_err(Error::Truncate)?;

    // Check truncation succeeded
    let actual_size = file.metadata().map_err(Error::GetMetadata)?.len() as usize;
    if actual_size != size {
        return Err(Error::Length {
            expected: size,
            actual: actual_size,
        });
    }

    // For extra safety, lets add seals to prevent futher modifications of
    // the file's size.
    let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;
    // If the previous memfd_create call worked, this means that the F_ADD_SEALS
    // option is supported and this should never fail.
    let rc = libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals);
    assert_eq!(rc, 0, "sealing failed on a memfd");

    // Retrieve the file descriptor
    let fd = file.into_raw_fd();

    // Memory map the file
    // The unsafety requirement here is ensured by the “Check truncation succeeded” section above
    let region = MmapRegion::new(size, fd)?;

    Ok(Shared { fd, region })
}

impl<T> Shared<T> {
    /// Creates and initializes a `Shared<T>` to value `val`.
    pub fn new(val: T) -> Result<Shared<T>, Error> {
        unsafe {
            // This is safe thanks to `T` being of size `size_of::<T>()`
            let res = create_shared::<T>(mem::size_of::<T>())?;

            // This is safe thanks to `res` not yet being shared as it has just been created
            ptr::write_volatile(res.region.ptr, val);

            Ok(res)
        }
    }
}

impl Shared<c_void> {
    // TODO: Make this able to return DSTs (when Rust will have a proper DST story)
    // This will also require implementing CoerceUnsized for proper !Sized handling
    /// Creates a `Shared<c_void>` that points to a memory region of size `size`, aligned to page
    /// boundary.
    ///
    /// Proper handling of the returned value is left up to the user.
    pub fn new_sized(size: usize) -> Result<Shared<c_void>, Error> {
        unsafe {
            create_shared(size)
            // `c_void` is a ZST so this is safe, and usage is up to the caller's judgement
        }
    }
}

impl<T> Shared<T> {
    /// Creates a `Shared<T>` from a pre-existing `fd`.
    ///
    /// # Safety
    ///
    /// This assumes that `fd` is a file descriptor that has been created by another instance of
    /// `Shared<T>`, and that it will never be used by anything else than `Shared<T>`. Note that
    /// not respecting this will at best do as bad as `std::mem::transmute`, and at worst end in
    /// tears.
    ///
    /// This also assumes that `fd` is not shared with another `Shared<T>`, as the output of
    /// `as_raw_fd()` is only a borrow. Do not forget to call `dup` before passing the file
    /// descriptor to `from_raw_fd` if you are not using `into_raw_fd` and have not passed the
    /// `RawFd` over a socket.
    ///
    /// Finally, this assumes that `fd` is not coming from another process, as otherwise for safety
    /// we would need a `T: ProcSync` bound.
    unsafe fn from_raw_fd_impl(fd: RawFd) -> Result<Shared<T>, Error> {
        // Retrieve the length of the file
        let mut statbuf = mem::zeroed::<libc::stat>();
        if libc::fstat(fd, &mut statbuf) != 0 {
            return Err(Error::GetMetadata(io::Error::last_os_error()));
        }
        assert_eq!(statbuf.st_mode & libc::S_IFMT, libc::S_IFREG);
        let size = statbuf.st_size as usize;

        // Memory map the file
        let region = MmapRegion::new(size, fd)?;

        Ok(Shared { fd, region })
    }

    /// Attempts to clone `data`.
    ///
    /// This will remap `data` at another location in memory, in addition to keeping `data` alive.
    pub fn try_clone(data: &Shared<T>) -> Result<Shared<T>, Error> {
        unsafe {
            let fd = libc::dup(data.as_raw_fd());
            if fd == -1 {
                return Err(Error::Dup(errno().into()));
            }

            // The unsafety requirements here are satisfied by the successful `dup` and the fact
            // that types are checked by the signature of this function.
            Self::from_raw_fd_impl(fd)
        }
    }

    /// Returns a mutable pointer to the data contained by `data`.
    ///
    /// Note that using this pointer needs the caller to handle synchronization themselves.
    pub fn as_mut_ptr(data: &Shared<T>) -> *mut T {
        data.region.ptr
    }

    /// Returns the size in bytes of the data contained by `data`.
    ///
    /// For a `Shared<T>` it is `mem::size_of<T>()` and for `Shared<c_void>` it
    /// is the size specified at creation.
    pub fn size(data: &Shared<T>) -> usize {
        data.region.size
    }
}

impl<T: ProcSync> Shared<T> {
    /// Creates a `Shared<T>` from pre-existing `fd`.
    ///
    /// # Safety
    ///
    /// This assumes that `fd` is a file descriptor that has been created by another instance of
    /// `Shared<T>`, and that it will never be used by anything else than `Shared<T>`. Note that
    /// not respecting this will at best do as bad as `std::mem::transmute`, and at worst end in
    /// tears.
    ///
    /// This also assumes that `fd` is not shared with another `Shared<T>`, as the output of
    /// `as_raw_fd()` is only a borrow. Do not forget to call `dup` before passing the file
    /// descriptor to `from_raw_fd` if you are not using `into_raw_fd` and have not passed the
    /// `RawFd` over a socket.
    pub unsafe fn from_raw_fd(fd: RawFd) -> Result<Shared<T>, Error> {
        Self::from_raw_fd_impl(fd)
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
        // For now, if `close` failed, we're ignoring it.
        // However it means that something happened, so we might some day want to change this, for
        // instance by logging something.
        // See also https://github.com/rust-lang/rfcs/pull/2677
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // This is safe thanks to the only way of sharing memory being through `from_raw_fd`, which
        // itself is available only when `T: ProcSync`. There is also no way to safely obtain an
        // &mut to the memory region. The only way to do it being by unsafely dereferencing a
        // pointer retrieved from the returned reference, which would be wildly unsafe and most
        // likely UB.
        // Note that on the other hand DerefMut would *not* be safe. Long live interior mutability.
        unsafe { &*self.region.ptr }
    }
}

impl<T> AsRawFd for Shared<T> {
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl<T> IntoRawFd for Shared<T> {
    fn into_raw_fd(mut self) -> RawFd {
        let res = self.fd;
        // Drop self.region without dropping self.
        // This is safe thanks to `mem::forget`'ing the partially-moved-out-of `self`
        unsafe {
            ptr::drop_in_place(&mut self.region);
            mem::forget(self);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    use std::{
        os::unix::net::UnixDatagram,
        process,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
        thread,
    };

    use sendfd::{RecvWithFd, SendWithFd};

    macro_rules! test_write_and_read {
        ($zone:expr, $size:expr) => {{
            let zone = $zone;
            let size = $size;
            let ptr_mut = &*zone as *const _ as *mut u8;
            for i in 0..size {
                ptr::write_volatile(ptr_mut.add(i), i as u8);
            }
            let ptr = &*zone as *const _ as *const u8;
            for i in 0..size {
                assert_eq!(ptr::read_volatile(ptr.add(i)), i as u8);
            }
        }};
    }

    #[test]
    fn new_sized_allocates_properly() {
        const SIZE: usize = 10 * 4 * 1024 + 1; // 10 pages (maybe) plus one byte
        let zone = Shared::new_sized(SIZE).unwrap();
        assert_eq!(Shared::size(&zone), SIZE);
        unsafe { test_write_and_read!(zone, SIZE) };
    }

    #[test]
    fn new_allocates_properly() {
        const SIZE: usize = 4 * 1024 - 1; // 1 page (maybe) minus one byte
        let zone = Shared::new([0u8; SIZE]).unwrap();
        assert_eq!(Shared::size(&zone), SIZE);
        unsafe { test_write_and_read!(zone, SIZE) };
    }

    macro_rules! test_sync_across_threads {
        ($zone_name:ident, $base_name:ident; $build_zone:stmt, $clone_zone:expr; $v:ident, $incr:expr, $decr:expr) => {{
            const $base_name: usize = 42;
            const INCR: usize = 9876500;
            const DECR: usize = 9012300;
            $build_zone
            let zone1 = $clone_zone;
            let zone2 = $clone_zone;
            let incr = thread::spawn(move || {
                let $v = &*zone1;
                for _ in 0..INCR {
                    $incr;
                }
            });
            let decr = thread::spawn(move || {
                let $v = &*zone2;
                for _ in 0..DECR {
                    $decr;
                }
            });
            incr.join().unwrap();
            decr.join().unwrap();
            assert_eq!($zone_name.load(Ordering::SeqCst), BASE + INCR - DECR);
        }};
    }

    macro_rules! test_sync_across_threads_arc {
        ($v:ident, $incr:expr, $decr:expr) => {
            test_sync_across_threads!(
                zone, BASE;
                let zone = Arc::new(Shared::new(AtomicUsize::new(BASE)).unwrap()),
                zone.clone();
                $v, $incr, $decr
            )
        }
    }

    #[test]
    #[should_panic]
    fn syncs_across_threads_test_can_fail() {
        test_sync_across_threads_arc!(
            v,
            v.store(
                v.load(Ordering::SeqCst).overflowing_add(1).0,
                Ordering::SeqCst
            ),
            v.store(
                v.load(Ordering::SeqCst).overflowing_sub(1).0,
                Ordering::SeqCst
            )
        );
    }

    #[test]
    fn syncs_across_threads() {
        test_sync_across_threads_arc!(
            v,
            v.fetch_add(1, Ordering::SeqCst),
            v.fetch_sub(1, Ordering::SeqCst)
        );
    }

    macro_rules! test_sync_across_threads_different_shared {
        ($v:ident, $incr:expr, $decr:expr) => {{
            test_sync_across_threads!(
                zone, BASE;
                let zone = Shared::new(AtomicUsize::new(BASE)).unwrap(),
                Shared::try_clone(&zone).unwrap();
                $v, $incr, $decr
            )
        }};
    }

    #[test]
    #[should_panic]
    fn syncs_across_threads_different_shared_can_fail() {
        test_sync_across_threads_different_shared!(
            v,
            v.store(
                v.load(Ordering::SeqCst).overflowing_add(1).0,
                Ordering::SeqCst
            ),
            v.store(
                v.load(Ordering::SeqCst).overflowing_sub(1).0,
                Ordering::SeqCst
            )
        );
    }

    #[test]
    fn syncs_across_threads_different_shared() {
        test_sync_across_threads_different_shared!(
            v,
            v.fetch_add(1, Ordering::SeqCst),
            v.fetch_sub(1, Ordering::SeqCst)
        );
    }

    macro_rules! test_sync_across_processes_with_fork {
        ($v:ident, $incr:expr, $decr:expr) => {{
            const BASE: usize = 1337;
            const INCR: usize = 8901200;
            const DECR: usize = 8765400;
            let zone = Shared::new((AtomicUsize::new(BASE), AtomicBool::new(false))).unwrap();
            let ($v, child_complete) = &*zone;
            let child = || {
                // In the child
                for _ in 0..INCR {
                    $incr;
                }
                child_complete.store(true, Ordering::SeqCst);
            };
            let parent = || {
                // In the parent
                for _ in 0..DECR {
                    $decr;
                }
                while !child_complete.load(Ordering::SeqCst) {
                    thread::yield_now();
                }
                assert_eq!(zone.0.load(Ordering::SeqCst), BASE + INCR - DECR);
            };
            unsafe {
                let pid = libc::fork();
                if pid == 0 {
                    child();
                    process::exit(0);
                } else {
                    parent();
                    libc::waitpid(pid, ptr::null_mut(), 0); // Reap child
                }
            }
        }};
    }

    #[test]
    #[should_panic]
    fn syncs_across_processes_with_fork_test_can_fail() {
        test_sync_across_processes_with_fork!(
            v,
            v.store(
                v.load(Ordering::SeqCst).overflowing_add(1).0,
                Ordering::SeqCst
            ),
            v.store(
                v.load(Ordering::SeqCst).overflowing_sub(1).0,
                Ordering::SeqCst
            )
        );
    }

    #[test]
    fn syncs_across_processes_with_fork() {
        test_sync_across_processes_with_fork!(
            v,
            v.fetch_add(1, Ordering::SeqCst),
            v.fetch_sub(1, Ordering::SeqCst)
        );
    }

    macro_rules! test_sync_across_processes_after_socket_send {
        ($v:ident, $incr:expr, $decr:expr) => {{
            const BASE: usize = 10;
            const INCR: usize = 9000000;
            const DECR: usize = 8000000;
            let (send, receive) = UnixDatagram::pair().unwrap();
            let child = || {
                // In the child
                let zone = Shared::new((AtomicUsize::new(BASE), AtomicBool::new(false))).unwrap();
                send.send_with_fd(&[], &[zone.as_raw_fd()])
                    .expect("send should succeed");
                let ($v, child_complete) = &*zone;
                for _ in 0..INCR {
                    $incr;
                }
                child_complete.store(true, Ordering::SeqCst);
            };
            let parent = || {
                // In the parent
                let mut fd = [0; 1];
                receive
                    .recv_with_fd(&mut [], &mut fd)
                    .expect("recv should succeed");
                let zone: Shared<(AtomicUsize, AtomicBool)> =
                    unsafe { Shared::from_raw_fd(fd[0]).unwrap() };
                let ($v, child_complete) = &*zone;
                for _ in 0..DECR {
                    $decr;
                }
                while !child_complete.load(Ordering::SeqCst) {
                    thread::yield_now();
                }
                assert_eq!($v.load(Ordering::SeqCst), BASE + INCR - DECR);
            };
            unsafe {
                let pid = libc::fork();
                if pid == 0 {
                    child();
                    process::exit(0);
                } else {
                    parent();
                    libc::waitpid(pid, ptr::null_mut(), 0); // Reap child
                }
            }
        }};
    }

    #[test]
    #[should_panic]
    fn syncs_across_processes_after_socket_send_test_can_fail() {
        test_sync_across_processes_after_socket_send!(
            v,
            v.store(
                v.load(Ordering::SeqCst).overflowing_add(1).0,
                Ordering::SeqCst
            ),
            v.store(
                v.load(Ordering::SeqCst).overflowing_sub(1).0,
                Ordering::SeqCst
            )
        );
    }

    #[test]
    fn syncs_across_processes_after_socket_send() {
        test_sync_across_processes_after_socket_send!(
            v,
            v.fetch_add(1, Ordering::SeqCst),
            v.fetch_sub(1, Ordering::SeqCst)
        );
    }
}
