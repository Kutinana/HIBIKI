# HIBIKI: Hierarchical Instruction Binding for Image Key-token Intervention

HIBIKI 是一个面向 ComfyUI 的自定义节点原型，目标是通过语法化区域提示减少多目标生成中的概念漂移。

## 语法说明

基本格式：

`全局描述, {局部文本 | x, y, width, height}`

示例：

`A beautiful street, {a man in a red shirt | 0,0,512,512}, {a woman in a blue dress | 512,0,512,512}`

说明：
- 全局描述：不在 `{}` 内的文本
- 局部描述：`{text | x,y,w,h}`
- 坐标单位：像素坐标（节点内部会按 8 对齐用于 area）

## 节点说明

节点名：`HIBIKI Regional Prompter`  
节点 ID：`HIBIKIRegionalPrompter`

输入：
- `text` (`STRING`)
- `clip` (`CLIP`)
- `strength` (`FLOAT`, 可选，默认 `1.0`)
- `set_cond_area` (`default` / `mask bounds`, 可选)
- `image_width` (`INT`, 可选，默认 `0`)
- `image_height` (`INT`, 可选，默认 `0`)
- `mask_blur_radius` (`INT`, 可选，默认 `0`)
- `mask_blur_sigma` (`FLOAT`, 可选，默认 `1.0`)

输出：
- `CONDITIONING`

关于 `image_width` / `image_height`：
- 两者都大于 0 时，节点会使用该尺寸生成画布（并按 8 对齐）
- 任一为 0 时，节点根据局部框自动推断画布尺寸

关于软边界：
- `mask_blur_radius > 0` 时启用高斯模糊（soft mask）
- 推荐起步：`mask_blur_radius=8`、`mask_blur_sigma=4.0`

## 接入 ComfyUI

1. 将本项目（或 `hibiki` 目录）放入 ComfyUI 的 `custom_nodes` 可加载位置。  
2. 重启 ComfyUI。  
3. 在节点搜索中输入 `HIBIKI Regional Prompter`。  
4. 将输出连接到 `KSampler` 的 `positive` 端进行采样测试。
