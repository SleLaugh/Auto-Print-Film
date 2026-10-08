#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

DICOM_ROOT="/home/peng/03_data/demo/Image/dicom"
RESULT_ROOT="/home/peng/03_data/demo/Image/result"
PYTHON_BIN=".venv/bin/python"
# 格式: "切割部位|状位|切割方式|层厚(mm)|层间距(mm)[|FillPercent]"
# 同一部位和状位可以配置多条，批量请求会按此处顺序全部保留。
# 整组统一居中/缩放；末尾 |80 表示本组目标整体范围占画面 80%，第六项填 0 或省略则不缩放。
# 本脚本不读取 .vscode/launch.json；层厚/层间距由第四、五项决定。
SLICE_TYPE_CONFIGS=(
    "颈椎|横状位|MPR|5|5|80"
    "颈椎|横状位|椎间隙|3|3|80"
    "颈椎|矢状位|MPR|5|5|80"
    "颈椎|冠状位|CPR|5|5|80"
    "胸椎|横状位|MPR|5|5|80"
    "胸椎|矢状位|MPR|5|5|80"
    "胸椎|冠状位|CPR|5|5|80"
    "腰椎|横状位|MPR|5|5|80"
    "腰椎|横状位|椎间隙|5|5|80"
    "腰椎|矢状位|MPR|5|5|80"
    "腰椎|冠状位|CPR|5|5|80"
    "肺部|横状位|MPR|5|5|80"
    "肺部|矢状位|MPR|5|5|80"
    "肺部|冠状位|MPR|5|5|80"
    "髋骨|横状位|MPR|5|5|80"
    "髋骨|矢状位|MPR|5|5|80"
    "髋骨|冠状位|MPR|5|5|80"
)
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

build_plane_configs() {
    local slice_part="$1"
    local plane item configured_part configured_plane configured_slice_type
    local configured_thickness configured_spacing configured_fill matched
    local config_args=()

    for plane in "横状位" "矢状位" "冠状位"; do
        matched=0
        for item in "${SLICE_TYPE_CONFIGS[@]}"; do
            IFS='|' read -r \
                configured_part \
                configured_plane \
                configured_slice_type \
                configured_thickness \
                configured_spacing \
                configured_fill <<< "$item"
            if [[ "$configured_part" == "$slice_part" && "$configured_plane" == "$plane" ]]; then
                config_args+=(
                    "$plane"
                    "$configured_slice_type"
                    "$configured_thickness"
                    "$configured_spacing"
                    "$configured_fill"
                )
                matched=1
            fi
        done
        if [[ "$matched" -eq 0 ]]; then
            printf '缺少切割方式配置: %s-%s\n' "$slice_part" "$plane" >&2
            return 1
        fi
    done

    "$PYTHON_BIN" -c '
import json
import sys

configs = []
for index in range(1, len(sys.argv), 5):
    plane, slice_type, slice_thickness, slice_spacing, fill_percent = sys.argv[index:index + 5]
    configs.append({
        "Plane": plane,
        "SliceThickness": float(slice_thickness),
        "SliceSpacing": float(slice_spacing),
        "WindowPresets": ["骨窗", "肺窗", "纵隔窗", "软组织窗"],
        "SliceType": slice_type,
    })
    if fill_percent:
        configs[-1]["FillPercent"] = float(fill_percent)
print(json.dumps(configs, ensure_ascii=False, separators=(",", ":")))
' "${config_args[@]}"
}

# 格式: "dicom目录名|result目录名|切割部位|扫描层厚(mm)"
CASES=(
    "颈1_1mm|颈椎_颈1_1mm|颈椎|1.0"
    "颈2_1mm|颈椎_颈2_1mm|颈椎|1.0"
    "颈3_1mm|颈椎_颈3_1mm|颈椎|1.0"
    "颈到盆1_1.5mm|胸椎_颈到盆1_1.5mm|胸椎|1.5"
    "颈到盆2_1.5mm|胸椎_颈到盆2_1.5mm|胸椎|1.5"
    "颈到腰1_1.5mm|胸椎_颈到腰1_1.5mm|胸椎|1.5"
    "颈到盆1_1.5mm|腰椎_颈到盆1_1.5mm|腰椎|1.5"
    "颈到盆2_1.5mm|腰椎_颈到盆2_1.5mm|腰椎|1.5"
    "颈到腰1_1.5mm|腰椎_颈到腰1_1.5mm|腰椎|1.5"
    "腰1_1mm|腰椎_腰1_1mm|腰椎|1.0"
    "腰2_1.5mm_歪|腰椎_腰2_1.5mm_歪|腰椎|1.5"
    "腰3_1mm|腰椎_腰3_1mm|腰椎|1.0"
    "肺1_1mm|肺部_肺1_1mm|肺部|1.0"
    "肺3_1mm|肺部_肺3_1mm|肺部|1.0"
    "颈到腰1_1.5mm|肺部_颈到腰1_1.5mm|肺部|1.5"
    "盆1_1.5mm|髋骨_盆1_1.5mm|髋骨|1.5"
    "盆2_1.5mm|髋骨_盆2_1.5mm|髋骨|1.5"
    "颈到盆1_1.5mm|髋骨_颈到盆1_1.5mm|髋骨|1.5"
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
    IFS='|' read -r dicom_name result_name slice_part scan_thickness <<< "${CASES[$index]}"
    input_dir="$DICOM_ROOT/$dicom_name"
    output_dir="$RESULT_ROOT/$result_name"

    printf '\n[%02d/%02d] %s -> %s (%s)\n' "$((index + 1))" "${#CASES[@]}" "$input_dir" "$output_dir" "$slice_part"

    if ! plane_configs="$(build_plane_configs "$slice_part")"; then
        failures+=("$dicom_name|$slice_part|missing slice type config")
        if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
            break
        fi
        continue
    fi

    if [[ ! -d "$input_dir" ]]; then
        printf '跳过：输入目录不存在: %s\n' "$input_dir" >&2
        failures+=("$dicom_name|$slice_part|missing input")
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
        "--slice_part" "$slice_part"
        "--scan_thickness" "$scan_thickness"
        "--plane_configs" "$plane_configs"
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
        failures+=("$dicom_name|$slice_part|cannot create log directory")
        if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
            break
        fi
        continue
    fi
    log_file="$log_dir/BatchLog.txt"

    printf '启动后台任务，日志: %s\n' "$log_file"
    "${cmd[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
    pid_labels+=("$dicom_name|$slice_part")
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
