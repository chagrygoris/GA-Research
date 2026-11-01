// Per-shape final results on SYMSOL I (10 % of the training data): the GATr flow with 49 pose harmonics (harm6) without
// and with nearest coupling (NC), and the GATr flow with the native STF track (optimist1938's run). Median = median
// rotation error to the nearest equivalent pose (degrees), acc@15 = share of test views with that error below 15
// degrees; the flows are decoded by the 32-sample geodesic medoid, and native STF also by a single flow sample
// (acc@15 only: the single-sample median error was not reported). harm6 + NC, 1 sample: one flow sample per view
// (scripts/symsol_spread.py --samples 1, seed 0) on the same checkpoint, kernel cheesewaffler/symsol-onesample-harm6-nc.
// Sources: harm6 = kernel log grig0ryz/symsol-gatr-spherecloud-harm6 and harm6 + NC = arynabiaroza/symsol-harm6-nc-l4x4
// (both in reports/symsol_stf/README.md, section 7); native STF L = 6 = kernel log saudabubin/symsol-native-stf6-no-nc-t4x2
// (--gatr_native_stf --gatr_native_stf_lmax 6 --gatr_native_stf_channels 8, no NC, 2 x T4).

#let acc(s) = if float(s) > 0.9 {
  table.cell(fill: rgb("#c8e6c9"))[#s]
} else [#s]

#figure(
  {
    set text(size: 9pt)
    table(
      columns: (auto, ..range(10).map(_ => 1fr)),
      align: (left, ..range(10).map(_ => right)),
      stroke: none,
      inset: (x: 3pt, y: 5pt),
      table.hline(),
      table.header(
        table.cell(rowspan: 2, align: left + bottom)[*Shape*],
        table.cell(colspan: 2, align: center)[*`harm6`*],
        table.cell(colspan: 2, align: center)[*`harm6` + NC*],
        table.cell(colspan: 2, align: center)[*`harm6` + NC, 1 сэмпл*],
        table.cell(colspan: 2, align: center)[*STF $L = 6$*],
        table.cell(colspan: 2, align: center)[*STF $L = 6$, 1 сэмпл*],
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
      [Tetrahedron], [41.91], acc("0.149"), [1.65], acc("0.966"), [1.83], acc("0.982"), [1.21], acc("0.923"), [---], acc("0.950"),
      [Cube],        [24.67], acc("0.287"), [2.29], acc("0.907"), [1.89], acc("0.963"), [3.35], acc("0.728"), [---], acc("0.894"),
      [Icosahedron], [31.19], acc("0.059"), [2.71], acc("0.909"), [2.49], acc("0.943"), [1.88], acc("0.845"), [---], acc("0.903"),
      [Cone],        [0.65], acc("1.000"), [0.55], acc("1.000"), [0.78], acc("1.000"), [0.63], acc("1.000"), [---], acc("1.000"),
      [Cylinder],    [0.98], acc("0.908"), [0.54], acc("0.985"), [0.69], acc("0.999"), [1.26], acc("0.969"), [---], acc("0.996"),
      table.hline(stroke: 0.5pt),
      [*Среднее*],   [19.88], acc("0.481"), [1.55], acc("0.954"), [1.54], acc("0.977"), [1.67], acc("0.893"), [---], acc("0.949"),
      table.hline(),
    )
  },
  caption: [
    Ошибки по фигурам на SYMSOL I (10 % обучающих данных): медиана угловой ошибки до ближайшей эквивалентной позы
    (в градусах) и $"acc@15"$.

    `harm6` --- GATr со сферическими гармониками $ell <= 6$ поворота как скалярными признаками; STF $L = 6$ ---
    GATr с отдельным эквивариантным треком симметричных бесследовых тензоров рангов $2..6$ (native STF). NC (nearest
    coupling) --- при обучении каждый начальный поворот $R_0$ переносится в ближайшую к нему из эквивалентных поз
    изображения, а не в случайно выбранную; колонки без «+ NC» обучены без него. `harm6` + NC: 4×L4 (DDP, SyncBN),
    один запуск; STF $L = 6$: 2×T4, один запуск. Все три потока, если не указано иное, декодируются геодезическим медоидом 32 сэмплов; «1 сэмпл» ---
    та же модель, декодированная одним сэмплом потока: для STF $L = 6$ медиана не приводилась. Все модели оценены
    на 5000 тестовых видах каждой фигуры. «Среднее»: среднее медиан по фигурам и $"acc@15"$ по всем видам.
  ],
) <tab:per-shape-harm6-stf>
