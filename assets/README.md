# 附件管理约定

本仓库不依赖图床。图片和小型附件保存在 Vault 内，由 Git 与笔记一起同步。

## 工作流

1. Obsidian 新粘贴的附件统一进入 `_inbox/`。
2. 整理笔记时，把定义、公式和结论转写为可搜索的 Markdown；视频中用于解释机制的关键画面保留为截图。
3. 重复、模糊或只用于临时提问的截图直接删除；有讲解价值的关键帧裁剪、重命名并长期保留。
4. Mermaid 只用于截图无法清楚表达、且图本身确实比原视频画面更直观的关系。
5. 在 Obsidian 文件管理器内完成重命名和移动，使链接自动更新。

## 目录与命名

```text
assets/
├── _inbox/
└── courses/
    └── stanford-cs336/
        ├── lecture-04/
        └── lecture-05/
```

课程视频截图使用“讲次 + 视频时间戳 + 内容”的名字：

```text
l05-15m51s-gpu-vs-tpu.png
l05-38m50s-mxfp8-block-scaling.png
l05-41m20s-mxfp8-transpose.webp
```

不要使用 `Pasted image 20260801140302.png` 这类只包含粘贴时间、无法识别内容的名字。

## 嵌入与来源

使用标准 Markdown 相对链接，不使用缺少路径的 Wiki 图片引用：

```markdown
![MXFP8 block scaling](../../assets/courses/stanford-cs336/lecture-05/l05-38m50s-mxfp8-block-scaling.png)

> 来源：[Lecture 5 38:50](https://www.youtube.com/watch?v=VIDEO_ID&t=2330s)
```

- 定义、公式和结论优先写成 Markdown/LaTeX，视频中的机制图保留关键帧并在下方补充文字解释。
- 文字/课件截图优先使用裁剪后的 PNG 或无损 WebP；照片才考虑有损 WebP/JPEG。
- 去除播放器控制栏、桌面边框等无关区域。
- 笔记与所引用的图片应在同一次 Git commit 中提交。
- 若仓库公开，只保留解释所必需的少量课程截图，并附原视频时间戳。
