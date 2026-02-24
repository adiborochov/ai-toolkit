#!/usr/bin/env python3
"""
Prepare before/after image pairs for processing.

1. Remove 'generated_' prefix from filenames in the after directory.
2. Verify both directories have matching filenames (ignoring extensions).
3. Check that each before/after pair has the same resolution.

Usage:
    python prepare_pairs.py <before_dir> <after_dir>
"""

import os
import sys
from PIL import Image


def remove_generated_prefix(after_dir):
    """Remove 'generated_' prefix from files in the after directory."""
    renamed = 0
    for fname in sorted(os.listdir(after_dir)):
        if fname.startswith("generated_"):
            old_path = os.path.join(after_dir, fname)
            new_name = fname[len("generated_"):]
            new_path = os.path.join(after_dir, new_name)
            os.rename(old_path, new_path)
            print(f"  Renamed: {fname} -> {new_name}")
            renamed += 1

    if renamed:
        print(f"\nRenamed {renamed} file(s).\n")
    else:
        print("No files with 'generated_' prefix found. Skipping rename step.\n")

    return renamed


def check_matching_filenames(before_dir, after_dir):
    """Check that both directories have the same basenames (without extension)."""
    def get_names(directory):
        names = {}
        for f in os.listdir(directory):
            if f.startswith("."):
                continue
            stem = os.path.splitext(f)[0]
            names[stem] = f
        return names

    before_names = get_names(before_dir)
    after_names = get_names(after_dir)

    only_before = sorted(set(before_names) - set(after_names))
    only_after = sorted(set(after_names) - set(before_names))
    common = sorted(set(before_names) & set(after_names))

    ok = True

    if only_before:
        print(f"Only in before ({len(only_before)}):")
        for name in only_before:
            print(f"  {before_names[name]}")
        ok = False

    if only_after:
        print(f"Only in after ({len(only_after)}):")
        for name in only_after:
            print(f"  {after_names[name]}")
        ok = False

    print(f"Matching pairs: {len(common)}")

    if ok:
        print("Filename check PASSED.\n")
    else:
        print("Filename check FAILED.\n")

    return ok, common, before_names, after_names


def check_matching_resolutions(before_dir, after_dir, common, before_names, after_names):
    """Check that each common pair has the same resolution."""
    mismatches = []

    for stem in common:
        before_path = os.path.join(before_dir, before_names[stem])
        after_path = os.path.join(after_dir, after_names[stem])

        before_size = Image.open(before_path).size
        after_size = Image.open(after_path).size

        if before_size != after_size:
            mismatches.append((stem, before_size, after_size))

    if mismatches:
        print(f"Resolution mismatches ({len(mismatches)}):")
        for stem, bsize, asize in mismatches:
            print(f"  {stem}: before={bsize[0]}x{bsize[1]}, after={asize[0]}x{asize[1]}")
        print("Resolution check FAILED.\n")
        return False
    else:
        print(f"All {len(common)} pairs have matching resolutions.")
        print("Resolution check PASSED.\n")
        return True


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <before_dir> <after_dir>")
        sys.exit(1)

    before_dir = os.path.abspath(sys.argv[1])
    after_dir = os.path.abspath(sys.argv[2])

    if not os.path.isdir(before_dir):
        print(f"Error: before directory not found: {before_dir}")
        sys.exit(1)
    if not os.path.isdir(after_dir):
        print(f"Error: after directory not found: {after_dir}")
        sys.exit(1)

    print(f"Before: {before_dir}")
    print(f"After:  {after_dir}\n")

    # Step 1: Remove generated_ prefix
    print("--- Step 1: Remove 'generated_' prefix ---")
    remove_generated_prefix(after_dir)

    # Step 2: Check matching filenames
    print("--- Step 2: Check matching filenames ---")
    names_ok, common, before_names, after_names = check_matching_filenames(before_dir, after_dir)

    # Step 3: Check matching resolutions
    print("--- Step 3: Check matching resolutions ---")
    if not common:
        print("No common pairs to check.\n")
        res_ok = False
    else:
        res_ok = check_matching_resolutions(before_dir, after_dir, common, before_names, after_names)

    # Summary
    if names_ok and res_ok:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
