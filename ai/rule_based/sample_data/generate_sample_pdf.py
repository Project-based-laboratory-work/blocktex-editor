"""
動作確認用のサンプルPDFを生成するスクリプト。
見出し・段落・表・図（矩形で代用）を含み、4-1-1のPDF解析検証と
4-1-2のルールベース分類のテスト入力として使う。

使い方:
    python sample_data/generate_sample_pdf.py
"""

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample.pdf")

# ReportLab組み込みの日本語CIDフォント（フォントファイルのダウンロード不要）
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))


class FigurePlaceholder(Flowable):
    """図の代わりに矩形を描画するだけのFlowable（画像素材を用意しなくても図オブジェクトを再現するため）。"""

    def __init__(self, width=80 * mm, height=40 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        self.canv.setFillColor(colors.lightgrey)
        self.canv.rect(0, 0, self.width, self.height, fill=1)


def build() -> None:
    styles = getSampleStyleSheet()
    # 日本語表示のため、既存スタイルのフォントをCIDフォントに差し替える
    styles["Title"].fontName = "HeiseiKakuGo-W5"
    styles["Heading1"].fontName = "HeiseiKakuGo-W5"
    styles["BodyText"].fontName = "HeiseiMin-W3"
    styles["Italic"].fontName = "HeiseiMin-W3"

    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4)

    story = [
        Paragraph("第1章 サンプル文書", styles["Title"]),
        Spacer(1, 10 * mm),
        Paragraph("1.1 はじめに", styles["Heading1"]),
        Paragraph(
            "これはPDF解析ライブラリの動作確認用に生成したサンプル文書である。"
            "見出し・段落・表・図の4種類のブロックを含み、"
            "ルールベース分類器の入力として利用する。",
            styles["BodyText"],
        ),
        Spacer(1, 6 * mm),
        Paragraph("1.2 表の例", styles["Heading1"]),
        Table(
            [
                ["項目", "値"],
                ["工数", "15時間"],
                ["担当", "加茂大地"],
            ],
            style=TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
                ]
            ),
        ),
        Spacer(1, 6 * mm),
        Paragraph("1.3 図の例", styles["Heading1"]),
        FigurePlaceholder(),
        Paragraph("図1: サンプル図（矩形で代用）", styles["Italic"]),
        Spacer(1, 6 * mm),
        Paragraph("1.4 まとめ", styles["Heading1"]),
        Paragraph(
            "以上のように、見出し・段落・表・図が混在する文書からブロック種別を"
            "判定できるかどうかを、このサンプルPDFで確認する。",
            styles["BodyText"],
        ),
    ]

    doc.build(story)
    print(f"generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
