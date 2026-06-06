#!/usr/bin/env bash
# Tiny fallback dataset used ONLY when BigVul ingestion is unavailable (fully
# offline first run). Train/val/test are DISJOINT samples — never copies.
# A model trained on this is a smoke-test artifact, not a usable classifier.
set -euo pipefail
BACKUP="data/backup_$(date +%Y%m%d_%H%M%S)"; mkdir -p data
backup(){ [[ -f "$1" ]] && { mkdir -p "$BACKUP"; mv -v "$1" "$BACKUP"/; } || true; }
for f in data/train.jsonl data/val.jsonl data/test.jsonl; do backup "$f"; done

cat > data/train.jsonl <<'JSONL'
{"code":"void g(char*s){char b[8];strcpy(b,s);}","label":1}
{"code":"#include <stdio.h>\nint f(char *fmt){char buf[128];sprintf(buf,fmt);return puts(buf);}","label":1}
{"code":"#include <stdlib.h>\nvoid h(){char* p=malloc(16);free(p);free(p);}","label":1}
{"code":"void r(){char b[4];gets(b);}","label":1}
{"code":"#include <string.h>\nvoid cat(char*d,const char*s){strcat(d,s);}","label":1}
{"code":"#include <stdlib.h>\nint* uaf(){int* p=malloc(4);free(p);return p;}","label":1}
{"code":"int ok(){char b[8]={0};return 0;}","label":0}
{"code":"#include <string.h>\nvoid safe(const char* s){char b[8];strncpy(b,s,sizeof(b)-1);b[7]='\\0';}","label":0}
{"code":"#include <stdio.h>\nvoid fsafe(const char* s){char b[128];snprintf(b,sizeof(b),\"%s\",s);}","label":0}
{"code":"int add(int a,int b){return a+b;}","label":0}
{"code":"#include <stdlib.h>\nvoid fine(){char* p=malloc(16);if(p){free(p);p=NULL;}}","label":0}
{"code":"#include <stdio.h>\nvoid rd(char* b,size_t n){fgets(b,(int)n,stdin);}","label":0}
JSONL

cat > data/val.jsonl <<'JSONL'
{"code":"void v(char*s){char d[16];strcpy(d,s);}","label":1}
{"code":"#include <stdio.h>\nvoid w(char* in){char b[32];sprintf(b,in);}","label":1}
{"code":"int mul(int a,int b){return a*b;}","label":0}
{"code":"#include <stdio.h>\nvoid vs(const char* s){char b[64];snprintf(b,sizeof(b),\"%s\",s);}","label":0}
JSONL

cat > data/test.jsonl <<'JSONL'
{"code":"void t(char*x){char y[8];strcat(y,x);}","label":1}
{"code":"#include <stdlib.h>\nvoid tf(){char* q=malloc(8);free(q);free(q);}","label":1}
{"code":"int sub(int a,int b){return a-b;}","label":0}
{"code":"#include <string.h>\nvoid ts(char*d,size_t n,const char*s){strncpy(d,s,n-1);d[n-1]='\\0';}","label":0}
JSONL

echo "[OK] Wrote bootstrap (smoke-test) datasets under ./data"; wc -l data/*.jsonl || true
