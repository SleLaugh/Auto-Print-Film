#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

DICOM_ROOT="/home/peng/03_data/demo/Image/dicom"
RESULT_ROOT="/home/peng/03_data/demo/Image/result"
PYTHON_BIN=".venv/bin/python"
PLANE_CONFIGS='[{"plane":"横状位","slice_thickness":5,"slice_spacing":5,"window_presets":["骨窗","肺窗","纵隔窗","软组织窗"],"line_extent_mode":"部位范围","SliceType":"MPR"},{"plane":"矢状位","slice_thickness":5,"slice_spacing":5,"window_presets":["骨窗","肺窗","纵隔窗","软组织窗"],"line_extent_mode":"部位范围","SliceType":"MPR"},{"plane":"冠状位","slice_thickness":5,"slice_spacing":5,"window_presets":["骨窗","肺窗","纵隔窗","软组织窗"],"line_extent_mode":"部位范围","SliceType":"MPR"}]'
# 中文请求值依赖 UTF-8 argv；显式开启可避免精简 Linux 环境使用 C/ASCII locale。
export PYTHONUTF8=1
DRY_RUN=0
STOP_ON_FAILURE=0
JOBS=1

usage() {
    printf 'Usage: %s [--dry-run] [--stop-on-failure] [--jobs N]\n' "$0"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --stop-on-failure)
            STOP_ON_FAILURE=1
            shift
            ;;
        --jobs|-j)
            if [[ $# -lt 2 ]] || ! is_positive_integer "$2"; then
                usage >&2
                printf -- '--jobs 必须是正整数\n' >&2
                exit 2
            fi
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            printf '未知参数: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

# 格式: "dicom目录名|result目录名|部位"
CASES=(
    "颈1_1mm|颈椎_颈1_1mm|颈椎"
    "颈2_1mm|颈椎_颈2_1mm|颈椎"
    "颈3_1mm|颈椎_颈3_1mm|颈椎"
    "颈到盆1_1.5mm|胸椎_颈到盆1_1.5mm|胸椎"
    "颈到盆2_1.5mm|胸椎_颈到盆2_1.5mm|胸椎"
    "颈到腰1_1.5mm|胸椎_颈到腰1_1.5mm|胸椎"
    "颈到盆1_1.5mm|腰椎_颈到盆1_1.5mm|腰椎"
    "颈到盆2_1.5mm|腰椎_颈到盆2_1.5mm|腰椎"
    "颈到腰1_1.5mm|腰椎_颈到腰1_1.5mm|腰椎"
    "腰1_1mm|腰椎_腰1_1mm|腰椎"
    "腰2_1.5mm_歪|腰椎_腰2_1.5mm_歪|腰椎"
    "腰3_1mm|腰椎_腰3_1mm|腰椎"
    "肺1_1mm|肺部_肺1_1mm|肺部"
    "肺3_1mm|肺部_肺3_1mm|肺部"
    "颈到腰1_1.5mm|肺部_颈到腰1_1.5mm|肺部"
    "盆1_1.5mm|髋骨_盆1_1.5mm|髋骨"
    "盆2_1.5mm|髋骨_盆2_1.5mm|髋骨"
    "颈到盆1_1.5mm|髋骨_颈到盆1_1.5mm|髋骨"
)

failures=()
pids=()
pid_labels=()
pid_logs=()

wait_for_slot() {
    while [[ "${#pids[@]}" -ge "$JOBS" ]]; do
        wait_for_one
    done
}

wait_for_one() {
    local pid label log_file
    pid="${pids[0]}"
    label="${pid_labels[0]}"
    log_file="${pid_logs[0]}"

    pids=("${pids[@]:1}")
    pid_labels=("${pid_labels[@]:1}")
    pid_logs=("${pid_logs[@]:1}")

    if ! wait "$pid"; then
        failures+=("$label|run failed|log=$log_file")
        if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
            wait_for_all
            return 1
        fi
    fi

    return 0
}

wait_for_all() {
    while [[ "${#pids[@]}" -gt 0 ]]; do
        wait_for_one || return 1
    done
}

for index in "${!CASES[@]}"; do
    IFS='|' read -r dicom_name result_name body_part <<< "${CASES[$index]}"
    input_dir="$DICOM_ROOT/$dicom_name"
    output_dir="$RESULT_ROOT/$result_name"

    printf '\n[%02d/%02d] %s -> %s (%s)\n' "$((index + 1))" "${#CASES[@]}" "$input_dir" "$output_dir" "$body_part"

    if [[ ! -d "$input_dir" ]]; then
        printf '跳过：输入目录不存在: %s\n' "$input_dir" >&2
        failures+=("$dicom_name|$body_part|missing input")
        if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
            break
        fi
        continue
    fi

    cmd=(
        "$PYTHON_BIN"
        "main.py"
        "-i" "$input_dir"
        "-o" "$output_dir"
        "--body_part" "$body_part"
        "--plane_configs" "$PLANE_CONFIGS"
    )

    printf '命令:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "$DRY_RUN" -eq 1 ]]; then
        continue
    fi

    wait_for_slot || break

    log_dir="$output_dir/log"
    if ! mkdir -p "$log_dir"; then
        failures+=("$dicom_name|$body_part|cannot create log directory")
        if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
            break
        fi
        continue
    fi
    log_file="$log_dir/BatchLog.txt"

    printf '启动后台任务，日志: %s\n' "$log_file"
    "${cmd[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
    pid_labels+=("$dicom_name|$body_part")
    pid_logs+=("$log_file")

    if [[ "$STOP_ON_FAILURE" -eq 1 ]] && [[ "${#failures[@]}" -gt 0 ]]; then
        break
    fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
    wait_for_all || true
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
    printf '\n失败/跳过用例:\n' >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

printf '\n批量测试完成，共 %d 条。\n' "${#CASES[@]}"
