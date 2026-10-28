// Per-shape final results on SYMSOL I (10 % of the training data): GATr flows with different pose / condition
// encodings, with and without nearest coupling (NC), and Image2Sphere. Median = median rotation error to the nearest
// equivalent pose (degrees), acc@15 = share of test views with that error below 15 degrees; the flows use the 32-sample
// geodesic medoid, Image2Sphere the argmax of its 1.875-degree SO(3) grid.
// Sources: Polar = frame_ch run (reports/pose_encoding_symmetry.md); Default = spherecloud GATr control and harm6 =
// the same with 49 pose harmonics (reports/i2s_symsol1_10pct.md, per shape from the kernel logs); harm6 + NC and
// STF L = 6 + NC: reports/symsol_stf/README.md, section 7; Image2Sphere: reports/i2s_symsol1_10pct.md.

#let acc(s) = if float(s) > 0.9 {
  table.cell(fill: rgb("#c8e6c9"))[#s]
} else [#s]

#figure(
  {
    set text(size: 9pt)
    table(
      columns: (auto, ..range(12).map(_ => 1fr)),
      align: (left, ..range(12).map(_ => right)),
      stroke: none,
      inset: (x: 3pt, y: 5pt),
      table.hline(),
      table.header(
        table.cell(rowspan: 2, align: left + bottom)[*Shape*],
        table.cell(colspan: 2, align: center)[*Polar*],
        table.cell(colspan: 2, align: center)[*Default*],
        table.cell(colspan: 2, align: center)[*`harm6`*],
        table.cell(colspan: 2, align: center)[*`harm6` + NC*],
        table.cell(colspan: 2, align: center)[*STF $L = 6$ + NC*],
        table.cell(colspan: 2, align: center)[*Image2Sphere*],
        table.hline(start: 1, end: 3, stroke: 0.5pt),
        table.hline(start: 3, end: 5, stroke: 0.5pt),
        table.hline(start: 5, end: 7, stroke: 0.5pt),
        table.hline(start: 7, end: 9, stroke: 0.5pt),
        table.hline(start: 9, end: 11, stroke: 0.5pt),
        table.hline(start: 11, end: 13, stroke: 0.5pt),
        [Median], [acc\@15],
        [Median], [acc\@15],
        [Median], [acc\@15],
        [Median], [acc\@15],
        [Median], [acc\@15],
        [Median], [acc\@15],
      ),
      table.hline(stroke: 0.5pt),
      [Tetrahedron], [54.45], acc("0.013"), [53.66], acc("0.012"), [41.91], acc("0.149"), [1.65], acc("0.966"), [8.42], acc("0.658"), [4.44], acc("1.000"),
      [Cube],        [42.28], acc("0.027"), [42.22], acc("0.025"), [24.67], acc("0.287"), [2.29], acc("0.907"), [26.50], acc("0.245"), [4.40], acc("1.000"),
      [Icosahedron], [31.51], acc("0.053"), [31.12], acc("0.059"), [31.19], acc("0.059"), [2.71], acc("0.909"), [31.60], acc("0.060"), [4.75], acc("1.000"),
      [Cone],        [0.66], acc("1.000"), [0.54], acc("1.000"), [0.65], acc("1.000"), [0.55], acc("1.000"), [0.57], acc("1.000"), [1.34], acc("1.000"),
      [Cylinder],    [1.38], acc("0.956"), [0.56], acc("0.994"), [0.98], acc("0.908"), [0.54], acc("0.985"), [0.68], acc("0.996"), [1.14], acc("1.000"),
      table.hline(stroke: 0.5pt),
      [*Среднее*],   [26.06], acc("0.410"), [25.62], acc("0.418"), [19.88], acc("0.481"), [1.55], acc("0.954"), [13.55], acc("0.592"), [3.21], acc("1.000"),
      table.hline(),
    )
  },
  caption: [
    Ошибки по фигурам на SYMSOL I (10 % обучающих данных): медиана угловой ошибки до ближайшей эквивалентной позы
    (в градусах) и $"acc@15"$. Решили перейти от замера log likelihood к $"acc@15"$ и медиане ошибки, поскольку
    подсчет функции плотности для Flow Matching вычислительно сложный, а результат несопоставим с подсчетом
    правдоподобия по HEALPix сетке Д. Клии и его коллег.

    Polar, Default --- разные варианты кодировки поворотов в GATr; `harm6` --- GATr со сферическими гармониками
    $ell <= 6$ как признаками; STF $L = 6$ --- GATr со вторым треком симметричных бесследовых тензоров рангов $2..6$
    (100 эпох, остальные потоки --- 40). NC (nearest coupling) --- при обучении каждый начальный поворот $R_0$
    переносится в ближайшую к нему из эквивалентных поз изображения, а не в случайно выбранную; колонки без «+ NC»
    обучены без него. `harm6` + NC: 4×L4 (DDP, SyncBN), один запуск. Default, `harm6` и `harm6` + NC оценены на
    всех 5000 тестовых видах каждой фигуры, Polar, STF и Image2Sphere --- на первых 1000. «Среднее»: среднее медиан
    по фигурам и $"acc@15"$ по всем видам.
  ],
) <tab:per-shape>
