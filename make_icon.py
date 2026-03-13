"""
生成 AudioTransfer.ico
设计：深色圆角方形背景 + 蓝色音频波形条 + 绿色传输箭头
运行：python make_icon.py
"""

from PIL import Image, ImageDraw, ImageFilter


def draw_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s    = size

    # ── 背景圆角矩形 ──────────────────────────────────────────
    r_bg = s // 5
    draw.rounded_rectangle([0, 0, s - 1, s - 1],
                            radius=r_bg,
                            fill=(30, 30, 46, 255))

    # 内部蓝色光晕（叠加一层半透明圆）
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    gd.ellipse([s * 0.15, s * 0.10, s * 0.85, s * 0.65],
               fill=(137, 180, 250, 28))
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img)

    # ── 波形条（5 根，不同高度） ───────────────────────────────
    bar_color   = (137, 180, 250, 255)   # #89b4fa 蓝
    heights_norm = [0.28, 0.50, 0.68, 0.50, 0.28]
    n      = len(heights_norm)
    bar_w  = s * 0.10
    gap    = s * 0.045
    total  = n * bar_w + (n - 1) * gap
    bx0    = (s - total) / 2
    cy     = s * 0.40          # 波形垂直中心（偏上，给箭头留空间）

    for i, hn in enumerate(heights_norm):
        h  = s * hn
        x0 = bx0 + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = cy - h / 2
        y1 = cy + h / 2
        draw.rounded_rectangle([x0, y0, x1, y1],
                                radius=bar_w / 2,
                                fill=bar_color)

    # ── 传输箭头（→） ────────────────────────────────────────
    ac     = (166, 227, 161, 220)        # #a6e3a1 绿
    ay     = s * 0.745                   # 箭头垂直中心
    aw     = s * 0.44                    # 箭头总宽
    ax0    = (s - aw) / 2
    ax1    = ax0 + aw
    sh_h   = s * 0.075                   # 杆粗细
    hd     = s * 0.16                    # 箭头头部尺寸

    # 杆
    draw.rounded_rectangle(
        [ax0, ay - sh_h / 2, ax1 - hd * 0.55, ay + sh_h / 2],
        radius=sh_h / 2,
        fill=ac
    )
    # 箭头头（三角形）
    draw.polygon(
        [(ax1,             ay),
         (ax1 - hd,        ay - hd * 0.72),
         (ax1 - hd,        ay + hd * 0.72)],
        fill=ac
    )

    return img


def main():
    sizes   = [16, 24, 32, 48, 64, 128, 256]
    frames  = [draw_icon(sz) for sz in sizes]

    # 保存为多分辨率 .ico
    frames[0].save(
        "AudioTransfer.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in sizes],
        append_images=frames[1:]
    )

    # 同时保存 256x256 PNG 预览
    frames[-1].save("AudioTransfer_preview.png")
    print("已生成 AudioTransfer.ico 和 AudioTransfer_preview.png")


if __name__ == "__main__":
    main()
