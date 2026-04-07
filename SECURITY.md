# Security

## Context

Sauhu is a **local desktop DICOM viewer** designed for single-user Linux workstations. It is not a server, not a web application, and not a multi-user system. The threat model and security considerations below reflect this context.

Sauhu is not a certified medical device. See the [README](README.md) disclaimer.

## Reporting Vulnerabilities

If you find a security issue, please email mikko@nyman.xyz. Do not open a public issue for security vulnerabilities.

## Threat Model

Sauhu runs as a local desktop application on the user's workstation. The primary attack surfaces are:

- **DICOM network (PACS)**: Sauhu connects to hospital PACS via standard DICOM protocol (C-FIND, C-MOVE, C-STORE)
- **Local file import**: Users can open DICOM files from disk
- **IPC socket**: External applications (e.g. dictation software) can send commands via Unix socket
- **Local database**: SQLite stores study metadata

The user's workstation is assumed to be a managed hospital device or a personal machine with standard security controls (disk encryption, user authentication, firewall).

## Known Issues and Status

### DICOM Network

| Issue | Status | Notes |
|-------|--------|-------|
| No TLS encryption | Open | Industry-wide: virtually no hospital PACS deploys DICOM TLS (Part 15). Sauhu follows the same protocol as every other DICOM client. PHI protection relies on network-level controls (hospital LAN, VPN). |
| Promiscuous SCP mode | Open | SCP accepts any AE title. Acceptable for a desktop viewer where the SCP runs only during active C-MOVE retrieval and is not a persistent service. AE title whitelist is a planned improvement. |
| SCP binds to 0.0.0.0 | Open | Should bind to localhost or a configured interface. The SCP is short-lived (active only during retrieval), but binding to all interfaces is unnecessarily broad. |

### Local Security

| Issue | Status | Notes |
|-------|--------|-------|
| Path traversal via Study UID | Open | Study UIDs are used in file paths with dots replaced by underscores, but `..` sequences are not explicitly blocked. Low risk: UIDs come from trusted PACS, not user input. Fix planned. |
| Mutex panic on poisoned lock | Open | Database mutex uses `.expect()` which panics on poisoned lock. Should return an error instead. |
| IPC socket permissions | Open | Socket created with default permissions. Should be 0600 (owner only). Low risk: `$XDG_RUNTIME_DIR` is already per-user on most Linux systems. |
| Database file permissions | Open | SQLite database uses default umask permissions. Should explicitly set 0600. |
| Clipboard not cleared | Open | Accession numbers read from clipboard are not cleared after use. Low risk for single-user workstation. |

### Resolved

| Issue | Version | Resolution |
|-------|---------|------------|
| DICOM parsing bounds | v0.1.0 | All array accesses have length validation |
| Sensitive data in repository | April 2026 | Hospital IPs, AE titles, patient IDs removed from code and git history |

## Compliance Notes

Sauhu is a local desktop viewer, not a healthcare information system. Regulatory compliance (GDPR, Finnish patient data law) is the responsibility of the infrastructure and organizational controls, not the viewer application itself. Relevant measures:

- PHI stays on the local workstation (no cloud transmission)
- PACS communication happens within hospital network or VPN
- Local storage uses the operating system's file permissions and disk encryption
- No user accounts or access control within sauhu (single-user application)

## Changelog

| Date | Changes |
|------|---------|
| 2026-01-10 | Initial security analysis |
| 2026-04-07 | Updated to reflect desktop application context, removed overstated compliance claims, purged sensitive data from repository |
