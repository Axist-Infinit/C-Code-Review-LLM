# review_inbox

Drop the C / C++ files you want reviewed in this directory, then run
`./ccr.sh` and pick **1) Review the inbox**.

Recognised extensions: .c .cpp .cc .cxx .c++ .h .hpp .hh .hxx .h++ .tcc .ipp .inl
Sub-directories are walked recursively. Anything else is ignored.

Reports are written to `scan_out/` (report.html, surface/surface_report.md,
*.sarif). Nothing here is sent anywhere — every lane runs locally.
