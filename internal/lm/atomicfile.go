package lm

import (
	"io"
	"os"
	"path/filepath"
)

// writeFileAtomic writes path through a uniquely named temporary file in the
// same directory, flushes it to disk, and renames it into place.
//
// Guarantees:
//   - A reader of path sees either the previous complete file or the new
//     complete file, never a partially written one, because the rename is
//     atomic within a directory.
//   - A failed or interrupted write leaves the previous file untouched and
//     removes its temporary file; temporary names are unique per process and
//     start with "." so concurrent saves cannot collide.
//   - The file contents are fsync'd before the rename, and the directory entry
//     is fsync'd afterwards (best effort: some filesystems reject directory
//     fsync). On ext4/xfs with default mount options a successful return means
//     the new checkpoint survives a power loss; on filesystems that reject
//     directory fsync, a rename that was already durable may still be lost.
func writeFileAtomic(path string, perm os.FileMode, write func(io.Writer) error) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-")
	if err != nil {
		return err
	}
	tmp := f.Name()
	committed := false
	defer func() {
		if !committed {
			f.Close()
			_ = os.Remove(tmp)
		}
	}()

	if err := f.Chmod(perm); err != nil {
		return err
	}
	if err := write(f); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	committed = true
	syncDir(dir)
	return nil
}

// syncDir makes the rename durable by flushing the directory entry. It is
// best effort: not every filesystem supports fsync on a directory.
func syncDir(dir string) {
	d, err := os.Open(dir)
	if err != nil {
		return
	}
	_ = d.Sync()
	_ = d.Close()
}
