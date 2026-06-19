#!/usr/bin/env bash
# Shared, ARCH-AWARE torch wheel resolution + install for the C code review
# pipeline. Sourced by setup_machine.sh, one_click_unlock_fetch_train_relock.sh,
# and install_torch_nightly_for_new_gpu.sh so wheel selection lives in ONE place.
#
# Why this exists: the DGX Spark is aarch64 Grace-Blackwell (GB10, sm_121). There
# is NO guaranteed *stable* cu130 aarch64 wheel on download.pytorch.org — the
# supported sources are the cu130 NIGHTLY channel or NVIDIA's NGC PyTorch SBSA
# (aarch64) container. Assuming a stable wheel and letting `pip install` fail
# under `set -e` aborts the entire setup with no actionable message. These
# helpers select the right channel per arch and FAIL SOFT (return non-zero,
# print a runbook pointer) so the caller can decide how to proceed.
#
# torch ONLY — torchvision/torchaudio are unused, and torchvision >=0.25 breaks
# the `datasets` torch formatter. Never install them here.
#
# Public functions:
#   ccr_detect_cuda_ver        -> echoes the CUDA version nvidia-smi reports (or "")
#   ccr_torch_channel ARCH CUDA STAGE -> echoes the wheel --index-url for arch/cuda
#                                        STAGE is "stable" or "nightly"
#   ccr_install_torch          -> resolves + installs torch for THIS host; returns
#                                 0 on success, non-zero (no abort) on failure.

ccr_t_say(){ printf "\033[1;36m[torch]\033[0m %s\n" "$*"; }
ccr_t_warn(){ printf "\033[1;33m[torch][warn]\033[0m %s\n" "$*"; }

ccr_detect_cuda_ver(){
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return 0; }
  # match both "CUDA Version: 12.4" and newer "CUDA UMD Version: 13.3" formats
  nvidia-smi 2>/dev/null | grep -oE "CUDA[ A-Za-z]* Version: [0-9]+\.[0-9]+" \
    | grep -oE "[0-9]+\.[0-9]+" | head -1 || true
}

# Echo the pip --index-url for (arch, cuda_ver, stage). stage=stable|nightly.
ccr_torch_channel(){
  local arch="$1" cuda="$2" stage="${3:-stable}"
  local base="https://download.pytorch.org/whl"
  [[ "$stage" == "nightly" ]] && base="https://download.pytorch.org/whl/nightly"

  # aarch64 (DGX Spark / Grace-Blackwell, sm_121): only cu130 carries sm_121
  # kernels, and on aarch64 it is a NIGHTLY-channel wheel (no stable cu130
  # aarch64 wheel is guaranteed). Always resolve to the cu130 channel and force
  # nightly regardless of the requested stage.
  if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
    echo "https://download.pytorch.org/whl/nightly/cu130"
    return 0
  fi

  # x86_64: map the driver's CUDA version to a wheel channel.
  case "$cuda" in
    13.*)              echo "$base/cu130" ;;
    12.8*|12.9*)       echo "$base/cu128" ;;
    12.5*|12.6*|12.7*) echo "$base/cu126" ;;
    12.4*)             echo "$base/cu124" ;;
    12.1*|12.2*|12.3*) echo "$base/cu121" ;;
    11.*)              echo "$base/cu118" ;;
    "")                echo "$base/cpu" ;;
    *)                 echo "$base/cu128" ;;   # unknown 12.x+: sane default
  esac
}

# Install torch for THIS host. Honors USE_GPU (default 1). On aarch64 uses the
# cu130 nightly channel with --pre. FAILS SOFT: returns non-zero (does not exit)
# so a missing/incompatible wheel does not abort the whole setup under set -e.
ccr_install_torch(){
  local use_gpu="${USE_GPU:-1}"
  local arch; arch="$(uname -m)"
  local pip_pkg="${CCR_TORCH_SPEC:-torch}"

  if [[ "$use_gpu" != "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    ccr_t_say "No GPU requested/visible; installing CPU torch."
    pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/cpu" $pip_pkg
    return $?
  fi

  local cuda; cuda="$(ccr_detect_cuda_ver)"
  if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
    local idx; idx="$(ccr_torch_channel "$arch" "$cuda" nightly)"
    ccr_t_say "aarch64 (Grace-Blackwell sm_121): installing cu130 NIGHTLY torch from $idx"
    ccr_t_warn "No stable cu130 aarch64 wheel is guaranteed; using the nightly channel."
    ccr_t_warn "If this fails, use the NVIDIA NGC PyTorch SBSA (aarch64) container instead."
    if ! pip install --pre --no-cache-dir --index-url "$idx" $pip_pkg; then
      ccr_t_warn "cu130 nightly aarch64 torch install FAILED."
      ccr_t_warn "Runbook: SETUP_SPARK.md -> 'torch on DGX Spark' (NGC SBSA container)."
      return 1
    fi
    return 0
  fi

  local idx; idx="$(ccr_torch_channel "$arch" "$cuda" stable)"
  ccr_t_say "x86_64 (CUDA '${cuda:-unknown}'): installing torch from $idx"
  if ! pip install --no-cache-dir --index-url "$idx" $pip_pkg; then
    ccr_t_warn "torch install from $idx FAILED."
    ccr_t_warn "Newer CUDA than the channel ships? See SETUP_4090.md -> 'torch on the 4090'."
    ccr_t_warn "Or run: ./install_torch_nightly_for_new_gpu.sh"
    return 1
  fi
  return 0
}
