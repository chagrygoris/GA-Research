// Per-shape final results on SYMSOL I (10 % of the training data) with nearest coupling (NC).
// Numbers: pose3d/reports/symsol_stf/README.md, section 7. Median = median rotation error to the nearest
// equivalent pose (degrees), acc@15 = share of test views with that error below 15 degrees; flows use the
// 32-sample geodesic medoid, Image2Sphere the argmax of its 1.875-degree SO(3) grid.

#let acc(s) = if float(s) > 0.9 {
  table.cell(fill: rgb("#c8e6c9"))[#s]
} else [#s]

#figure(
  table(
    columns: (auto, ..range(10).map(_ => 1fr)),
    align: (left, ..range(10).map(_ => right)),
    stroke: none,
    inset: (x: 3.5pt, y: 5pt),
    table.hline(),
    table.header(
      table.cell(rowspan: 2, align: left + bottom)[*Shape*],
      table.cell(colspan: 2, align: center)[*`harm6`*],
      table.cell(colspan: 2, align: center)[*`harm6` + NC, 4×L4*],
      table.cell(colspan: 2, align: center)[*`harm6` + NC, RTX*],
      table.cell(colspan: 2, align: center)[*STF $L = 6$ + NC, 100 эпох*],
      table.cell(colspan: 2, align: center)[*Image2Sphere*],
      table.hline(start: 1, end: 3, stroke: 0.5pt),
      table.hline(start: 3, end: 5, stroke: 0.5pt),
      table.hline(start: 5, end: 7, stroke: 0.5pt),
      table.hline(start: 7, end: 9, stroke: 0.5pt),
      table.hline(start: 9, end: 11, stroke: 0.5pt),
      [Median], [acc\@15],
      [Median], [acc\@15],
      [Median], [acc\@15],
      [Median], [acc\@15],
      [Median], [acc\@15],
    ),
    table.hline(stroke: 0.5pt),
    [Tetrahedron], [41.91], acc("0.149"), [1.65], acc("0.966"), [9.48], acc("0.677"), [8.42], acc("0.658"), [4.44], acc("1.000"),
    [Cube],        [24.67], acc("0.287"), [2.29], acc("0.907"), [7.40], acc("0.738"), [26.50], acc("0.245"), [4.40], acc("1.000"),
    [Icosahedron], [31.19], acc("0.059"), [2.71], acc("0.909"), [7.03], acc("0.782"), [31.60], acc("0.060"), [4.75], acc("1.000"),
    [Cone],        [0.65], acc("1.000"), [0.55], acc("1.000"), [0.50], acc("1.000"), [0.57], acc("1.000"), [1.34], acc("1.000"),
    [Cylinder],    [0.98], acc("0.908"), [0.54], acc("0.985"), [0.52], acc("1.000"), [0.68], acc("0.996"), [1.14], acc("1.000"),
    table.hline(stroke: 0.5pt),
    [*Среднее*],   [19.88], acc("0.481"), [1.55], acc("0.954"), [4.99], acc("0.839"), [13.55], acc("0.592"), [3.21], acc("1.000"),
    table.hline(),
  ),
  caption: [
    Ошибки по фигурам на SYMSOL I (10 % обучающих данных): медиана угловой ошибки до ближайшей эквивалентной позы
    (в градусах) и $"acc@15"$. NC (nearest coupling) --- при обучении каждый начальный поворот $R_0$ переносится в
    ближайшую к нему из эквивалентных поз изображения, а не в случайно выбранную.

    `harm6` --- GATr со сферическими гармониками $ell <= 6$ как признаками, без NC; `harm6` + NC --- тот же рецепт
    с NC (40 эпох), на 4×L4 (DDP, SyncBN) и на одной RTX Pro 6000; разница между ними не объяснена (один запуск на
    каждую конфигурацию). STF $L = 6$ --- второй трек симметричных бесследовых тензоров рангов $2..6$. Потоки оценены
    на всех 5000 тестовых видах каждой фигуры, Image2Sphere --- на первых 1000. «Среднее»: среднее медиан по
    фигурам и $"acc@15"$ по всем видам.
  ],
) <tab:per-shape-nc>
