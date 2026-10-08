"""Open a folder in the file manager with several fallbacks, and log why each one failed (Windows diagnostics).

On a normal Windows session the first method works. On a machine where the shell call is refused
("[WinError 5] Access is denied") the other methods are tried in turn, and the process state (restricted token,
integrity level, job object, session) is written to the log once, so the cause can be read from app.log.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_diagnosed = False


def process_diagnostics() -> str:
    """One line describing the security state of this process (Windows only; never raises)."""
    if sys.platform != "win32":
        return "not Windows"
    try:
        import ctypes
        from ctypes import wintypes

        kernel, advapi = ctypes.windll.kernel32, ctypes.windll.advapi32
        parts: list[str] = []
        token = wintypes.HANDLE()
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi.IsTokenRestricted.argtypes = [wintypes.HANDLE]
        advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.POINTER(wintypes.DWORD)]
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        advapi.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        process = kernel.GetCurrentProcess()
        if advapi.OpenProcessToken(process, 0x0008, ctypes.byref(token)):       # TOKEN_QUERY
            parts.append(f"restricted_token={bool(advapi.IsTokenRestricted(token))}")
            size = wintypes.DWORD(0)
            advapi.GetTokenInformation(token, 25, None, 0, ctypes.byref(size))                      # TokenIntegrityLevel
            buffer = ctypes.create_string_buffer(size.value)
            if size.value and advapi.GetTokenInformation(token, 25, buffer, size, ctypes.byref(size)):
                advapi.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
                advapi.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
                sid = ctypes.c_void_p.from_buffer(buffer).value
                count = advapi.GetSidSubAuthorityCount(ctypes.c_void_p(sid))[0]
                parts.append(f"integrity_rid={advapi.GetSidSubAuthority(ctypes.c_void_p(sid), count - 1)[0]:#x}")
            kernel.CloseHandle(token)
        in_job = wintypes.BOOL()
        if kernel.IsProcessInJob(process, None, ctypes.byref(in_job)):
            parts.append(f"in_job={bool(in_job.value)}")
        session = wintypes.DWORD()
        if kernel.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
            parts.append(f"session={session.value}")
        parts.append(f"SystemRoot={'set' if os.environ.get('SystemRoot') else 'MISSING'}")
        return ", ".join(parts)
    except Exception as exc:       # noqa: BLE001 - diagnostics must never break the caller
        return f"diagnostics failed: {exc}"


def _shell_execute_explore(path: str) -> None:
    import ctypes

    result = ctypes.windll.shell32.ShellExecuteW(None, "explore", path, None, None, 1)
    if result <= 32:
        raise OSError(f"ShellExecuteW returned {result}")


def _cmd_start(path: str) -> None:
    """`cmd /c start` with the Windows error dialog suppressed, so a refused start never shows a system message."""
    import ctypes

    old = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x8000)       # SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX
    try:
        subprocess.run(["cmd.exe", "/c", "start", "", path], check=True, timeout=10,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise OSError("cmd start timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise OSError(f"cmd start exited with {exc.returncode}") from exc
    finally:
        ctypes.windll.kernel32.SetErrorMode(old)


def open_folder(path: str | Path, qt_open: Callable[[str], bool] | None = None) -> bool:
    """Try every method; True as soon as one works. Failures are logged (path only, never a secret)."""
    global _diagnosed
    folder = str(path)
    methods: list[tuple[str, Callable[[], object]]] = []
    if qt_open is not None:
        methods.append(("Qt", lambda: qt_open(folder) or (_ for _ in ()).throw(OSError("QDesktopServices refused"))))
    if sys.platform == "win32":
        methods += [("startfile", lambda: os.startfile(folder)),        # noqa: S606 - a folder the user chose
                    ("ShellExecute explore", lambda: _shell_execute_explore(folder)),
                    ("cmd start", lambda: _cmd_start(folder))]
    elif sys.platform == "darwin":
        methods.append(("open", lambda: subprocess.run(["open", folder], check=True, timeout=10)))
    else:
        methods.append(("xdg-open", lambda: subprocess.run(["xdg-open", folder], check=True, timeout=10)))
    for name, attempt in methods:
        try:
            attempt()
            return True
        except Exception as exc:           # noqa: BLE001 - every method may fail in its own way
            log.warning("Opening %s with %s failed: %s", folder, name, exc)
    if not _diagnosed:
        _diagnosed = True
        log.warning("Process state: %s", process_diagnostics())
    return False
