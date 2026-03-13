#!/bin/sh
# Use meson setup (not deprecated "meson build") for clean build
set -e
rm -rf build
meson setup build
ninja -C build "$@"
sudo ninja -C build install
gst-inspect-1.0 rknnfilter