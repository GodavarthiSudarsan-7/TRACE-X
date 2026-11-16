"""Safe, static Windows PE analysis for TRACE-X.

This module parses uploaded bytes in memory. It NEVER executes uploaded files.
Requires: pefile
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".exe", ".dll"}
SUSPICIOUS_SECTION_NAMES = {
    ".upx0", ".upx1", "upx0", "upx1", ".aspack", ".adata",
    ".themida", ".vmp0", ".vmp1",
}
SUSPICIOUS_IMPORT_TERMS = (
    "virtualalloc", "virtualprotect", "writeprocessmemory",
    "createremotethread", "winexec", "shellexecute",
    "urldownloadtofile", "internetopen", "httpsendrequest",
    "regsetvalue", "createservice", "openprocess",
)

def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return round(-sum((count / length) * math.log2(count / length)
                      for count in counts.values()), 4)

def _extract_strings(data: bytes, minimum: int = 5, limit: int = 300) -> list[str]:
    """Extract printable ASCII and UTF-16LE strings for triage, not deobfuscation."""
    found: list[str] = []
    seen: set[str] = set()
    patterns = (
        re.compile(rb"[\x20-\x7e]{%d,}" % minimum),
        re.compile(rb"(?:[\x20-\x7e]\x00){%d,}" % minimum),
    )
    for pattern in patterns:
        for match in pattern.finditer(data):
            raw = match.group(0)
            try:
                value = raw.decode("ascii" if pattern is patterns[0] else "utf-16le", errors="ignore")
            except Exception:
                continue
            value = value.strip()
            if value and value not in seen:
                seen.add(value)
                found.append(value[:500])
                if len(found) >= limit:
                    return found
    return found

def analyze_pe_bytes(data: bytes, filename: str) -> dict[str, Any]:
    """Analyze a Windows PE file from bytes and return JSON-serializable findings."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Only Windows .exe and .dll files are supported in this phase.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 10 MB analysis limit.")
    if not data.startswith(b"MZ"):
        raise ValueError("The file does not have a valid Windows MZ header.")

    try:
        import pefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pefile'. Install it with: pip install pefile") from exc

    try:
        pe = pefile.PE(data=data, fast_load=False)
        pe.parse_data_directories()
    except Exception as exc:
        raise ValueError(f"Unable to parse this file as a valid PE: {exc}") from exc

    sections = []
    warnings = []
    for section in pe.sections:
        name = section.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        raw = section.get_data() or b""
        entropy = _entropy(raw)
        virtual_size = int(section.Misc_VirtualSize)
        raw_size = int(section.SizeOfRawData)
        item = {
            "name": name,
            "virtual_address": hex(int(section.VirtualAddress)),
            "virtual_size": virtual_size,
            "raw_size": raw_size,
            "entropy": entropy,
            "characteristics": hex(int(section.Characteristics)),
        }
        sections.append(item)
        if name.lower() in SUSPICIOUS_SECTION_NAMES:
            warnings.append(f"Unusual or packer-associated section name: {name}")
        if entropy >= 7.2 and raw_size >= 1024:
            warnings.append(f"High-entropy section detected: {name} (entropy {entropy})")

    imports = []
    suspicious_imports = []
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode("utf-8", errors="replace") if entry.dll else "unknown"
            for imp in entry.imports:
                function = imp.name.decode("utf-8", errors="replace") if imp.name else f"ordinal_{imp.ordinal}"
                full_name = f"{dll}!{function}"
                imports.append(full_name)
                if any(term in function.lower() for term in SUSPICIOUS_IMPORT_TERMS):
                    suspicious_imports.append(full_name)

    exports = []
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols[:500]:
            name = symbol.name.decode("utf-8", errors="replace") if symbol.name else None
            exports.append({"name": name, "ordinal": int(symbol.ordinal),
                            "address": hex(int(symbol.address))})

    machine_map = {0x14c: "x86 (32-bit)", 0x8664: "x64 (64-bit)", 0x1c0: "ARM",
                   0xaa64: "ARM64"}
    machine = int(pe.FILE_HEADER.Machine)
    timestamp = int(pe.FILE_HEADER.TimeDateStamp)
    try:
        compile_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        compile_time = "Unavailable"

    if suspicious_imports:
        warnings.append("Potentially sensitive API imports are present; manual review is required.")
    if not sections:
        warnings.append("No PE sections were found.")

    result = {
        "analysis_type": "Static Windows PE analysis",
        "execution_performed": False,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "filename": Path(filename).name,
        "file_size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "file_type": "Windows Portable Executable",
        "architecture": machine_map.get(machine, f"Unknown (0x{machine:04x})"),
        "is_dll": bool(pe.FILE_HEADER.Characteristics & 0x2000),
        "entry_point": hex(int(pe.OPTIONAL_HEADER.AddressOfEntryPoint)),
        "image_base": hex(int(pe.OPTIONAL_HEADER.ImageBase)),
        "subsystem": int(pe.OPTIONAL_HEADER.Subsystem),
        "compile_timestamp_utc": compile_time,
        "sections": sections,
        "imports_count": len(imports),
        "imports": imports[:1000],
        "suspicious_imports": suspicious_imports,
        "exports": exports,
        "strings": _extract_strings(data),
        "warnings": sorted(set(warnings)),
        "interpretation": (
            "Static triage indicators only. High entropy, section names, and API imports "
            "are not proof of malware. Validate findings with trusted sources and, if "
            "needed, controlled sandbox analysis."
        ),
    }
    pe.close()
    return result
