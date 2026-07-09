# 阶段三定位与回正占位实现计划

## 目标

阶段三先稳定完成“定位”：

- 根据 `body_part + cut_mode` 找到本次任务需要的 label。
- 输出每个目标 label 的 bbox、centroid 和体素数量。
- 输出本次目标整体 bbox，作为后续映射和人工验证依据。
- 输出研发期 target mask overlay，用来在 ITK-SNAP 中直接核对目标体素位置。
- 输出研发期 label mask overlay，用来在 ITK-SNAP 中区分各椎体自己的范围。
- 默认保留“回正”这个业务步骤，但回正算法状态为 TODO。
- 默认不生成 `raw_corrected.nii.gz` / `seg_corrected.nii.gz`，后续阶段继续读取原始 raw/seg。

## 设计约束

- `correction_requested` 默认是 `true`，表示流程上确实有回正阶段。
- `correction_status` 默认是 `"todo"`，表示算法未实现。
- `correction_applied` 默认是 `false`，表示没有真的改写体数据。
- `correction_angle_degrees` 默认是 `0.0`，避免旧 PCA 角度被误当作可执行回正角。
- `raw_for_reconstruction_path` 指向原始 `raw_0000.nii.gz`。
- `seg_for_reconstruction_path` 指向原始 `seg.nii.gz`。
- `debug_bbox_overlay_path` 是轴对齐外接盒，只表达 min/max 范围。
- `debug_target_mask_overlay_path` 是真实目标体素，人工验证定位偏不偏优先看它。
- `debug_label_mask_overlay_path` 是按原始模型 label id 区分的目标体素，用于看各椎体区间。
- 旧字段 `corrected_raw_path` / `corrected_seg_path` 只做兼容；未回正时也指向原始路径。
- 后续真回正必须是整套 3D 体数据的一次刚性变换，不能逐张切片独立处理。

## 已完成

- [x] 从 `dataset.json` 读取 label 名称和 id。
- [x] 优先使用分割输出目录旁边的 sidecar `dataset.json`，避免读错模型 label 表。
- [x] 支持脊柱整切范围展开，例如 C1 到 C7 会覆盖中间所有颈椎 label。
- [x] 输出每个目标 label 的 bbox、centroid 和 voxel_count。
- [x] 输出本次目标整体 `correction_scope_bbox`。
- [x] 输出研发期 `localization_bbox_overlay.nii.gz`、`localization_target_mask_overlay.nii.gz`
      和 `localization_label_mask_overlay.nii.gz`。
- [x] 阶段三 metadata 写入 `raw_for_reconstruction_path` 和 `seg_for_reconstruction_path`。
- [x] 阶段五优先读取 `raw_for_reconstruction_path`，不再直接依赖旧 `corrected_raw_path`。
- [x] 默认回正流程改为 TODO/no-op，不再自动生成 corrected NIfTI。
- [x] 保留低层旋转 helper 作为后续算法实验入口，但默认主流程不调用。

## 暂不做

- [ ] 不实现自动回正角度计算。
- [ ] 不实现真实图像回正。
- [ ] 不生成 corrected NIfTI。
- [ ] 不实现矢状位/冠状位回正。
- [ ] 不实现 MPR/DICOM 输出。

## 后续回正算法必须重新验证

后面如果重新实现回正，不能直接复用旧的
`vertebrae_label_angle_aggregation` 作为可执行角度。它在真实颈椎 CT 上会把
椎体走形、椎体形态、金属和局部结构误判为病人体位旋转。

新的回正算法至少需要：

- 有明确的体数据级别角度来源。
- 对正位样例给出“不回正”结论。
- 对局部结构倾斜和整体体位倾斜做区分。
- 低置信度时拒绝回正。
- 回正前输出诊断指标，方便人工验证。

## 验证命令

```bash
python -m unittest tests/test_stage3_target_axial_correction.py -v
python -m unittest discover -s tests -v
python -B -c "import ast, pathlib; files=[pathlib.Path('main.py')]+list(pathlib.Path('src').glob('**/*.py'))+list(pathlib.Path('tests').glob('**/*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'parsed {len(files)} python files')"
python -m json.tool src/config/localization_config.json
python -m json.tool src/config/nnuNet_config.json
```
