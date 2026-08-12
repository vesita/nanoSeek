# Windows 交叉编译：zig 用 gnu 目标，不要用 msvc

> 背景：给推理运行时打 Windows 包（`package.py --target windows`）。本机 Linux，用 zig + cargo-zigbuild 交叉编译。首次选了 `x86_64-pc-windows-msvc` 目标，被 tokenizers 的 C++ 依赖卡死。2026-08-12。

## 现象

```
error occurred in cc-rs: command did not execute successfully ...
cargo:warning=In file included from .../ziglang/lib/libcxx/include/__new/placement_new_delete.h:20:
cargo:warning=   20 | #  include <new.h>
cargo:warning=fatal error: 'new.h' file not found
```

编译 `tokenizers` 的 C++ 依赖 `esaxx-rs` 时，找不到 `<new.h>`。

## 根因

`<new.h>` 是 **MSVC 专属的 CRT 头**（VS 里才带）。`-msvc` ABI 下 zig 的 libc++ 设了 `_LIBCPP_ABI_VCRUNTIME`，于是：

```c
#if defined(_LIBCPP_ABI_VCRUNTIME)
#  include <new.h>      // ← MSVC CRT 头，zig 不内置
#else
  inline void* operator new(...) { ... }   // ← 自带实现，gnu 走这条
#endif
```

zig 交叉编译 **不内置 MSVC 头**（那要装 Visual Studio / Windows SDK），所以一碰 msvc ABI 的 C++ 就缺头。

## 修复

目标从 `x86_64-pc-windows-msvc` 改成 **`x86_64-pc-windows-gnu`**，一次通过。zig 完整内置 mingw-w64 的 CRT 头（含 `new.h`），编出的 .exe 是正常 Windows 程序，照常跑。

`package.py` 里 `TARGET = 'x86_64-pc-windows-gnu'`，并把原因写进了注释。

## 怎么避免

- 交叉编译 Windows **默认用 gnu 目标**，别用 msvc——除非打算装 Visual Studio。
- 工具链就位方式：`cargo install cargo-zigbuild` + `uv pip install ziglang`（zig 二进制在 `venv/lib/*/site-packages/ziglang/zig`，用 `CARGO_ZIGBUILD_ZIG_PATH` 环境变量指给 cargo-zigbuild，`package.py` 已自动探测）。
- 如果 msvc 目标真的必要，需要把 MSVC 头路径喂给 zig（复杂，本项目不需要）。
