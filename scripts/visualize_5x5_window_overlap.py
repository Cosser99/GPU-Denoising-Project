#!/usr/bin/env python3
"""Create an SVG showing the overlap between two consecutive filter windows.

The default figure shows a 5x5 filter window sliding one pixel to the right.
Cells reused by the next iteration are highlighted.

Iteration 1 is labeled vertically on the left.
Iteration 2 is labeled vertically on the right.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


CELL = 48

# Horizontal space reserved for the iteration labels.
LABEL_MARGIN = 55


def window_cells(top: int, left: int, kernel: int) -> set[tuple[int, int]]:
    return {
        (row, col)
        for row in range(top, top + kernel)
        for col in range(left, left + kernel)
    }


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1,
    rx: float = 0,
    opacity: float = 1,
    dash: str | None = None,
) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'width="{width:.1f}"',
        f'height="{height:.1f}"',
        f'fill="{fill}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
        f'rx="{rx}"',
        f'opacity="{opacity}"',
    ]

    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')

    return f"<rect {' '.join(attrs)} />"


def cell_xy(row: int, col: int) -> tuple[float, float]:
    return LABEL_MARGIN + col * CELL, row * CELL


def base_pixel_color(row: int, col: int) -> str:
    red = 226 - ((row * 15 + col * 7) % 38)
    green = 232 - ((row * 9 + col * 13) % 42)
    blue = 238 - ((row * 11 + col * 5) % 34)

    return f"#{red:02x}{green:02x}{blue:02x}"


def validate_window(
    rows: int,
    cols: int,
    kernel: int,
    stride: int,
    start_row: int,
    start_col: int,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]]:

    if rows <= 0 or cols <= 0:
        raise SystemExit("rows e cols devono essere positivi.")

    if kernel <= 0:
        raise SystemExit("kernel deve essere positivo.")

    if stride <= 0:
        raise SystemExit("stride deve essere positivo.")

    current = (start_row, start_col)

    if direction == "right":
        next_window = (start_row, start_col + stride)
    else:
        next_window = (start_row + stride, start_col)

    for label, (top, left) in (
        ("current", current),
        ("next", next_window),
    ):
        if (
            top < 0
            or left < 0
            or top + kernel > rows
            or left + kernel > cols
        ):
            raise SystemExit(
                f"La finestra {label} esce dalla griglia: "
                f"top={top}, left={left}, "
                f"kernel={kernel}, rows={rows}, cols={cols}."
            )

    return current, next_window


def build_svg(
    rows: int,
    cols: int,
    kernel: int,
    stride: int,
    start_row: int,
    start_col: int,
    direction: str,
) -> str:

    current, next_window = validate_window(
        rows,
        cols,
        kernel,
        stride,
        start_row,
        start_col,
        direction,
    )

    current_cells = window_cells(*current, kernel)
    next_cells = window_cells(*next_window, kernel)

    shared_cells = current_cells & next_cells
    current_only = current_cells - next_cells
    next_only = next_cells - current_cells

    grid_width = cols * CELL
    grid_height = rows * CELL

    # Only the space required by the two vertical labels.
    width = LABEL_MARGIN + grid_width + LABEL_MARGIN
    height = grid_height

    current_color = "#60a5fa"
    shared_color = "#86efac"
    next_color = "#fb923c"

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',

        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',

        # Transparent background.
    ]

    # ------------------------------------------------------------------
    # Grid cells
    # ------------------------------------------------------------------

    for row in range(rows):
        for col in range(cols):

            x, y = cell_xy(row, col)

            fill = base_pixel_color(row, col)

            if (row, col) in shared_cells:
                fill = shared_color

            elif (row, col) in current_only:
                fill = current_color

            elif (row, col) in next_only:
                fill = next_color

            # No white borders between cells.
            parts.append(
                rect(
                    x,
                    y,
                    CELL,
                    CELL,
                    fill,
                    "#ffffff",
                    0.5
                )
            )

    # ------------------------------------------------------------------
    # Outer grid border
    # ------------------------------------------------------------------

    parts.append(
        rect(
            LABEL_MARGIN,
            0,
            grid_width,
            grid_height,
            "none",
            "#334155",
            2,
        )
    )

    # ------------------------------------------------------------------
    # Window outlines
    # ------------------------------------------------------------------

    def window_rect(
        top_left: tuple[int, int],
        color: str,
        dash: str | None,
    ) -> None:

        top, left = top_left

        x, y = cell_xy(top, left)

        parts.append(
            rect(
                x,
                y,
                kernel * CELL,
                kernel * CELL,
                "none",
                color,
                5,
                dash=dash,
            )
        )

    # Iteration 1: dashed blue
    window_rect(
        current,
        "#2563eb",
        "12 8",
    )

    # Iteration 2: solid orange
    window_rect(
        next_window,
        "#ea580c",
        None,
    )

    # ------------------------------------------------------------------
    # Iteration labels
    # ------------------------------------------------------------------

    def window_label(
        top_left: tuple[int, int],
        text: str,
        color: str,
        side: str,
    ) -> None:

        top, left = top_left

        x, y = cell_xy(top, left)

        center_y = y + (kernel * CELL) / 2

        if side == "left":

            label_x = x - 18
            rotation = -90

        else:

            label_x = x + kernel * CELL + 18
            rotation = 90

        parts.append(
            f'<text '
            f'x="{label_x:.1f}" '
            f'y="{center_y:.1f}" '
            f'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="22" '
            f'font-weight="700" '
            f'fill="{color}" '
            f'text-anchor="middle" '
            f'transform="rotate('
            f'{rotation} '
            f'{label_x:.1f} '
            f'{center_y:.1f})">'
            f'{text}'
            f'</text>'
        )

    window_label(
        current,
        "Iteration 1",
        "#2563eb",
        "left",
    )

    window_label(
        next_window,
        "Iteration 2",
        "#ea580c",
        "right",
    )

    # ------------------------------------------------------------------
    # Finish SVG
    # ------------------------------------------------------------------

    parts.append("</svg>")

    return "\n".join(parts)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Genera una figura SVG che mostra l'overlap "
            "tra due finestre consecutive."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "reports/visuals/5x5_window_overlap.svg"
        ),
        help="Percorso del file SVG da generare.",
    )

    parser.add_argument(
        "--png-output",
        type=Path,
        help=(
            "Percorso opzionale della PNG da esportare. "
            "Richiede rsvg-convert."
        ),
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=8,
        help="Numero di righe della griglia.",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=11,
        help="Numero di colonne della griglia.",
    )

    parser.add_argument(
        "--kernel",
        type=int,
        default=5,
        help="Dimensione della finestra quadrata.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Spostamento tra due iterazioni.",
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="Riga iniziale della prima finestra.",
    )

    parser.add_argument(
        "--start-col",
        type=int,
        default=2,
        help="Colonna iniziale della prima finestra.",
    )

    parser.add_argument(
        "--direction",
        choices=("right", "down"),
        default="right",
        help=(
            "Direzione di scorrimento "
            "della seconda finestra."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    svg = build_svg(
        rows=args.rows,
        cols=args.cols,
        kernel=args.kernel,
        stride=args.stride,
        start_row=args.start_row,
        start_col=args.start_col,
        direction=args.direction,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        svg,
        encoding="utf-8",
    )

    print(
        f"Figura generata: {args.output}"
    )

    # ------------------------------------------------------------------
    # Optional PNG export
    # ------------------------------------------------------------------

    if args.png_output:

        converter = shutil.which(
            "rsvg-convert"
        )

        if not converter:
            raise SystemExit(
                "Impossibile esportare la PNG: "
                "installa rsvg-convert oppure "
                "apri direttamente l'SVG."
            )

        args.png_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        subprocess.run(
            [
                converter,
                str(args.output),
                "-o",
                str(args.png_output),
            ],
            check=True,
        )

        print(
            f"PNG generata: {args.png_output}"
        )


if __name__ == "__main__":
    main()