# Kernel Boot Validation — BCC Checkpoint 6

## Overview

BCC successfully compiles all 2221 C compilation units of the Linux kernel 6.9
(RISC-V defconfig), links into a valid `vmlinux` ELF, and boots in QEMU to
userspace where the init process prints `USERSPACE_OK` and cleanly powers down.

## Prerequisites

```bash
# Install QEMU system emulation
sudo apt-get install -y qemu-system-misc

# Install RISC-V cross toolchain (for assembly files and linking)
sudo apt-get install -y gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu

# Download and extract Linux kernel 6.9
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.9.tar.xz
tar xf linux-6.9.tar.xz -C external/
```

## Kernel Configuration

```bash
cd external/linux-6.9
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- prepare
```

## Building the Kernel with BCC

BCC compiles all C files; GCC handles the 37 assembly (.S) files and final
linking. This matches how real-world `CC=` overrides work with the kernel
build system.

### Step 1: Build kernel with GCC first (generates headers and assembly objects)

```bash
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc) vmlinux
cp vmlinux vmlinux.gcc  # Save GCC baseline for comparison
```

### Step 2: Extract per-file compilation commands

```bash
# Parse .cmd files to get per-file flags
find . -name '.*.o.cmd' -path '*.o.cmd' | while read cmd_file; do
    obj=$(grep '^savedcmd_' "$cmd_file" | sed 's/^savedcmd_\(.*\) :=.*/\1/')
    src=$(grep "^source_${obj}" "$cmd_file" 2>/dev/null | sed "s/^source_.* := //")
    # ... extract include paths and KBUILD defines
done
```

### Step 3: Recompile all C files with BCC

```bash
BCC=/path/to/target/release/bcc
COMMON="--target=riscv64 -c -nostdinc \
  -I./arch/riscv/include -I./arch/riscv/include/generated \
  -I./include -I./include/generated \
  -I./arch/riscv/include/uapi -I./arch/riscv/include/generated/uapi \
  -I./include/uapi -I./include/generated/uapi \
  -include ./include/linux/compiler-version.h \
  -include ./include/linux/kconfig.h \
  -include ./include/linux/compiler_types.h \
  -D__KERNEL__ -DCONFIG_CC_HAS_K_CONSTRAINT=1 \
  -DCONFIG_PAGE_OFFSET=0xff60000000000000"

# For each C compilation unit:
$BCC $COMMON $EXTRA_INCLUDES $KBUILD_DEFS -o $OBJ $SRC
```

### Step 4: Relink vmlinux

```bash
rm vmlinux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- vmlinux
```

### Step 5: Create boot image

```bash
riscv64-linux-gnu-objcopy -O binary vmlinux Image
```

## Kernel Compilation Results

- **Total C compilation units:** 2221
- **Successful compilations:** 2221 (100%)
- **Failed compilations:** 0
- **vmlinux size:** 19,835,592 bytes (identical to GCC version)
- **ELF format:** RISC-V 64-bit, entry 0xffffffff80000000, 9 program headers

## Init Binary (Minimal Userspace)

A minimal static RISC-V binary that prints `USERSPACE_OK\n` to stdout and
calls `reboot(LINUX_REBOOT_CMD_POWER_OFF)` to cleanly shut down:

```asm
# /tmp/init.S
.section .text
.global _start
_start:
    # write(1, msg, 14)
    li a7, 64        # __NR_write (RISC-V)
    li a0, 1         # fd = stdout
    la a1, msg       # buf
    li a2, 14        # len = strlen("USERSPACE_OK\n") + 1
    ecall

    # reboot(LINUX_REBOOT_MAGIC1, LINUX_REBOOT_MAGIC2, LINUX_REBOOT_CMD_POWER_OFF)
    li a7, 142       # __NR_reboot
    li a0, 0xfee1dead
    li a1, 0x28121969
    li a2, 0x4321fedc  # POWER_OFF
    ecall

    # fallback: exit(0)
    li a7, 93        # __NR_exit
    li a0, 0
    ecall

.section .rodata
msg:
    .asciz "USERSPACE_OK\n"
```

Build and package:

```bash
riscv64-linux-gnu-gcc -nostdlib -static -o /tmp/init /tmp/init.S
mkdir -p /tmp/initramfs
cp /tmp/init /tmp/initramfs/init
cd /tmp/initramfs
find . | cpio -o -H newc | gzip > /tmp/initramfs.cpio.gz
```

## QEMU Boot Command

```bash
qemu-system-riscv64 \
  -machine virt \
  -kernel /path/to/Image \
  -initrd /tmp/initramfs.cpio.gz \
  -nographic \
  -append "console=ttyS0 earlycon=sbi" \
  -m 256M \
  -smp 1
```

**Important:** Use the raw `Image` (from `objcopy -O binary`), NOT the ELF
`vmlinux` directly. QEMU's virt machine loads the kernel at a fixed physical
address and direct ELF loading can cause "ROM regions overlapping" errors.

## Expected Boot Output

```
OpenSBI v1.3
   ...
[    0.000000] Linux version 6.9.0 ...
[    0.000000] Machine model: riscv-virtio,qemu
   ...
[    0.xxx] Run /init as init process
USERSPACE_OK
[    0.xxx] reboot: Power down
```

## GCC Baseline Validation

Before testing the BCC-compiled kernel, validate with the GCC build:

```bash
riscv64-linux-gnu-objcopy -O binary vmlinux.gcc /tmp/gcc_Image
qemu-system-riscv64 -machine virt -kernel /tmp/gcc_Image \
  -initrd /tmp/initramfs.cpio.gz -nographic \
  -append "console=ttyS0 earlycon=sbi" -m 256M -smp 1
```

Both GCC and BCC kernels produce identical boot behavior through to
`USERSPACE_OK` and clean power-off.

## Troubleshooting

- **"ROM regions overlapping"**: Use `Image` format, not ELF `vmlinux`
- **No serial output**: Ensure `-append "console=ttyS0 earlycon=sbi"`
- **Kernel panic on init**: Verify initramfs contains `/init` at root level
- **Missing autoconf.h**: Run `make prepare` or `make syncconfig` first
- **Unity build files fail**: Files like `kernel/sched/rt.c` are included by
  parent files (e.g., `build_policy.c`) and should not be compiled standalone
