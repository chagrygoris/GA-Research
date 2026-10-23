# GATr на flash-clifford: можно ли и что это даст (spike)

Статус: **spike, 2026-10-13.** Ответ на вопрос «можно ли перевезти GATr из SOTA-рецепта `origin/main` (9d47e0b: GATr vector field + warp/synth + EMA, batch 64) на Triton-ядра [flash-clifford](https://github.com/maxxxzdn/flash-clifford) и какой будет прирост скорости обучения и инференса». Код замеров throwaway, в репо не входит. Сырые числа лежат в `bench_results.json` рядом (ключи `runN/stage/variant`).

Все замеры сделаны на Kaggle RTX PRO 6000 Blackwell (sm_120, аккаунт avgurpetrovich) с torch 2.10.0+cu128 и triton 3.6.0, на модели из `origin/main` с дефолтным конфигом. Обучение: batch 64, n_time_samples 8, fp32, AdamW, EMA, GradScaler, всё как в `trainer.py`. Веса и картинки случайные, dataloader не участвует. Время — медиана 30 итераций после 10 прогревочных, по CUDA events. Между запусками eager-шаг гуляет на ±1.5% (175.4 / 180.3 / 180.0 ms).

## 0. Вывод

1. **Целиком перевезти GATr на flash-clifford нельзя: перевозить почти нечего.** Библиотека написана под CGENN-слои в Cl(2,0) и Cl(3,0). Для PGA G(3,0,1) в ней есть ровно одна операция, `gp_pga`: голое геометрическое произведение, не экспортированное из `ops/`. EquiLinear (9 Pin-базисных карт), equivariant join, equivariant layer norm и геометрического attention там нет.
2. **`gp_pga` встаёт в GATr drop-in и даёт точно ту же математику.** Порядок базиса и знаки совпадают с таблицей GATr во всех 256 парах блейдов. Forward и градиенты совпадают на уровне fp32-шума. Ядро работает на sm_120. Сама операция ускоряется в 18× (fwd+bwd 2.06 → 0.11 ms), но gp — лишь 9% времени vector field. В итоге **шаг обучения быстрее на 3.5–4.8%**, final eval (32 сэмпла) на 11%, а per-epoch eval не ускоряется вовсе.
3. **GATr упирается не в геометрическое произведение.** 49% времени vector field уходит на backward `EquiLinear`: einsum с 9 базисными картами плохо раскладывается в backward. Ещё 17% съедает attention, 12% layer norm, 10% join. В inference при batch 64 GATr упирается в CPU: GPU занят 2.7 ms из 9.6 ms. Причина — ~820 запусков ядер и скрытые блокирующие H2D-копии индексов (раздел 4).
4. **Большая часть выигрыша достаётся без новых Triton-ядер, одним PyTorch.** Математика и fp32 при этом сохраняются:
   - обучение: dense EquiLinear + flash gp + `torch.compile` дают шаг **180 → 132 ms (1.36×)**. Это ~4.5 ч на 100 эпох вместо 6.1 ч;
   - per-epoch inference: CUDA graph vector field даёт **182 → 80–89 ms на батч (2.0–2.3×)**;
   - final eval (32 сэмпла): **−14% в fp32**, −24% с TF32.
5. **Потолок полного переноса на Triton:** даже при нулевом времени vector field шаг не опустится ниже 98 ms (ResNet-101 89 ms + optimizer/EMA 9 ms), то есть максимум 1.84×. Реалистичный потолок полного ручного порта — около 1.7×, и `torch.compile` уже забирает ~60% этого запаса. Остаток (132 → ~105 ms) требует писать с нуля 4–5 Triton-ядер с backward под G(3,0,1), в первую очередь join, norm и fused bilinear.

Рекомендация: **flash-clifford не тащить.** У библиотеки нет лицензии, проверка градиентов в её тестах — no-op (`grad - grad`), и для PGA она даёт ~4%. Брать дешёвые правки из раздела 6. Вопрос о своих G(3,0,1)-ядрах поднимать только тогда, когда 1.36× станет мало.

## 1. Что умеет flash-clifford (commit e52775c, 2026-05-02)

| что | Cl(2,0) / Cl(3,0) | G(3,0,1) PGA |
|---|---|---|
| fused GELU → weighted GP → RMS-norm (CGENN-стиль, fwd+bwd в Triton) | есть | нет |
| fully-connected GP (`fc_*`, Triton + `torch.bmm`) | есть | нет |
| голое геометрическое произведение | — | `ops/p3m0n1.py: gp_pga`, fwd+bwd, не экспортировано |
| линейный слой | grade-wise `torch.bmm` (не ядро) | нет |
| join / outer / inner product, attention, Pin-EquiLinear | нет | нет |

Особенности:
- Раскладка компонентная, `(16, N, C)` contiguous; у GATr раскладка `(..., C, 16)`.
- `num_pos` и `num_channels` объявлены `tl.constexpr`, поэтому на каждую новую форму ядро перекомпилируется.
- Блок 4×128 с 16 warps подобран под RTX A4500, autotune нет.
- Тестировался только fp32. В bf16 у нас вышла относительная ошибка 1.3%.
- Лицензии нет, pip-пакета нет.

## 2. Где GATr тратит время (eager, fp32)

Шаг обучения 175–180 ms раскладывается так:

| часть | ms | доля |
|---|---|---|
| ResNet-101 + CGENN condition head, fwd+bwd | 86–89 | ~49% |
| **GATr vector field, fwd+bwd** (512 последовательностей × 66 токенов) | **82–84** | **~47%** |
| clip + AdamW + EMA | 9 | ~5% |

Для сверки с реальными прогонами: шаг 180 ms × 1141 шаг ≈ 205 s, а наблюдалось 214–222 s на эпоху, включая валидацию. CGENN-рецепт на той же RTX шёл 132–137 s на эпоху. Значит, 80–85 s эпохи — это цена GATr.

Vector field упирается в GPU: из 82 ms ядра заняты 75.6 ms, 3287 ядер на fwd+bwd. Каждый перехваченный вызов прогнали по отдельности с теми же входами:

| семейство ops (вызовов на forward) | fwd, ms | fwd+bwd, ms | доля fwd+bwd |
|---|---|---|---|
| EquiLinear (42) | 5.3 | **44.1** | **49%** |
| attention core (4) | 3.2 | 15.2 | 17% |
| equivariant layer norm (8) | 2.2 | 11.0 | 12% |
| equivariant join (4) | 2.4 | 9.0 | 10% |
| **геометрическое произведение (4)** — то, что покрывает flash-clifford | 2.2 | **8.5** | **9%** |
| gated GELU (4) | 0.4 | 1.9 | 2% |

По Амдалю ускорение одного gp в 18× даёт 4.2% на шаг. Измерено 3.5–4.8%.

## 3. Обучение: варианты (шаг на 64 картинки, ms)

Варианты от `eager` до `dense+flash+graphsafe+CUDA graph` и оба `torch.compile` считают ту же математику в fp32. Эквивалентность проверена так:
- на GPU: выход vector field отличается на ≤1e-9 при масштабе 0.013;
- на CPU, против fp64-эталона: у eager и у всех вариантов относительная ошибка выхода и всех невырожденных градиентов одинакова, ≤3.3e-6;
- dense EquiLinear отдельно, в fp64: ≤1e-14 по выходу и всем градиентам, на 7 конфигурациях каналов.

Вырожденными считались градиенты bias ключей attention: их истинное значение ~1e-23 из-за инвариантности softmax к сдвигу, поэтому относительная ошибка там бессмысленна.

| вариант | VF fwd+bwd | шаг | Δ шага |
|---|---|---|---|
| eager GATr 1.4.2 (как в main) | 84.3 | 180.0 | — |
| flash `gp_pga` вместо gp | 76.7 | 173.7 | −3.5% |
| graphsafe GATr (раздел 4, без H2D-синков) | 78.1 | 172.8 | −4.0% |
| dense EquiLinear (один matmul, обучаемый) | 68.4 | 164.4 | −8.7% |
| dense + flash | 62.0 | 158.3 | −12.1% |
| dense + flash + graphsafe | 56.3 | 149.7 | −16.8% |
| … + CUDA graph vector field (`make_graphed_callables`) | 51.8 | 145.7 | −19.1% |
| `torch.compile` (eager GATr) | 56.9 | 150.7 | −16.3% |
| **`torch.compile` (dense + flash)** | **38.1** | **132.4** | **−26.5%** |

Варианты, которые меняют численность и требуют прогона с проверкой точности:

| вариант | VF fwd+bwd | шаг | Δ шага |
|---|---|---|---|
| TF32 matmul, eager | 61.2 | 157.0 | −12.8% |
| TF32, dense + flash | 52.0 | 147.0 | −18.3% |
| bf16 autocast на весь шаг (включая ResNet), dense + flash | — | 115.8 | −35.7% |

`torch.compile` (inductor, режим по умолчанию) компилируется 27–36 s. Чтобы он работал, нужен `gatr.utils.einsum.enable_cached_einsum(False)`. Режим `reduce-overhead` на этой модели патологичен: velocity 150–500 ms, перезапись графов. Использовать его не надо.

## 4. Inference (батч 64 картинки, 20 шагов Эйлера, ms)

Per-epoch валидация предсказывает с 1 сэмплом: GATr видит 64 последовательности, GPU почти простаивает. Final eval предсказывает с 32 сэмплами: GATr видит 2048 последовательностей и загружает GPU. Backbone + condition head занимают 29.5 ms на батч в обоих режимах.

| вариант | 1 сэмпл: velocity / predict | 32 сэмпла: velocity / predict |
|---|---|---|
| eager (main) | 9.6 / 182 | 56.8 / 1163 |
| flash gp | 11.8 / 223 | 50.4 / 1035 |
| dense + flash | 11.9 / 229 | 51.9 / 1067 |
| frozen dense (`gatr.utils.compile_linear`) + flash | 7.0 / 157 | 50.4 / 1039 |
| graphsafe eager | 8.9 / 169 | 55.9 / 1151 |
| `torch.compile` eager / dense+flash | 6.1 / 129 · 5.9 / 122 | 48.4 / 1031 · 39.2 / 972 |
| **CUDA graph VF, eager** | **2.7 / 89** | 53.9 / 1113 |
| **CUDA graph VF, frozen dense + flash** | **2.2 / 80** | 48.5 / 1005 |
| … + TF32 | 1.4 / 63 | **42.6 / 887** |

При 1 сэмпле eager упирается в CPU: GPU занят 2.7 ms из 9.6 ms, ~820 ядер. Поэтому flash и dense там даже вредят: они добавляют транспозиции и ядра. Помогает только меньшее число запусков, то есть CUDA graph или compile.

Почему CUDA graph не захватывался «из коробки». В GATr 1.4.2 есть блокирующие H2D-копии: индексация Python-списками (`x[..., [0]]`, `reference[..., [14]]`, `tri[..., [3]]`, `_TRIVECTOR_IDX`, `_INNER_PRODUCT_*_IDX`) и CPU-константа `INNER_PRODUCT_INDICES.to(x.device)`, которая копируется на каждом вызове. «Graphsafe»-патч убирает их без смены значений: срезы вместо списков, константы сразу на GPU. Тогда захват работает, а eager ускоряется сам по себе на ~4–7%. Такая же копия есть в `clifford.CliffordAlgebra.embed` (`mv[..., tensor_index]` с CPU-индексом), поэтому в граф берётся только vector field, а эмбеддинг ротора остаётся eager.

Сколько это даёт на реальных прогонах. Final eval двух моделей (EMA и raw) в логе GATr-прогона занимал 48 s из 6.1 ч, так что inference почти не влияет на длительность обучения. Inference-ускорение нужно для оценки и деплоя, а также для per-epoch валидации: там ~2× на predict-части.

## 5. Полный перенос на Triton: потолок и цена

- **Потолок:** при нулевом vector field шаг ≥ 98 ms, то есть ≤1.84× к 180 ms.
- **Реалистичная оценка.** Минимум, к которому можно прийти:
  - EquiLinear — плотный GEMM, ~0.2 TFLOP на fwd+bwd;
  - остальные GA-ops — memory-bound, ~2.7 GB трафика при 1.44 TB/s.
  
  Это ~5–8 ms на vector field, шаг ~105 ms, **~1.7×**. С `torch.compile` (dense + flash) уже есть 1.36×.
- **Что пришлось бы написать** под G(3,0,1) (fwd+bwd, с собственными тестами градиентов, раз у upstream они пустые):
  1. join — хардкод 16×16 → 16 по образцу `gp_pga` плюс скаляр e0123 reference;
  2. equivariant layer norm (inner product по 8 компонентам без e0, редукция по каналам);
  3. fused bilinear block: 4 linear → gp ‖ join → concat → linear;
  4. сборка q/k-признаков attention (inner-product компоненты и distance-признаки тривекторов) перед SDPA.
  
  EquiLinear своё ядро не нужно: это обычный cuBLAS GEMM после свёртки 9 базисных карт.
- Почти то же, что flash-ядро, даёт хардкод-gp на чистом torch под `torch.compile`: fwd 0.041 ms против 0.044 ms, fwd+bwd 0.18 ms против 0.11 ms. Сама по себе библиотека тут не нужна, важна только идея хардкода таблицы Кэли.

## 6. Что делать (порядок по цене)

1. **Dense EquiLinear.** ~40 строк, monkeypatch `EquiLinear.forward`: свернуть 9 карт и s2mvs/mvs2s/s2s в один `F.linear`. Математика та же, ~−9% шага. Блочная раскладка повторяет `gatr.utils.compile_linear.compile_equi_linear`, только веса остаются обучаемыми.
2. **Graphsafe GATr.** 5 замен «список → срез» и 4 константы на GPU. Около −4% шага, −7% на inference, плюс возможность CUDA graph.
3. **`torch.compile(model.vector_field)`** с `enable_cached_einsum(False)`, вместе с п.1 и gp (flash или хардкод на torch): **1.36× на шаг**. Нужен флаг вида `--compile_vf`.
4. **Inference:** frozen dense (`compile_equi_linear_submodules`) + CUDA graph vector field дают per-epoch predict 2.3×, final eval −14%.
5. Только после прогона с проверкой 8.95°: TF32 (−13…−18%) и bf16 autocast (−36% на шаг; ResNet в bf16 даёт основную часть выигрыша).
6. Свои G(3,0,1) Triton-ядра (раздел 5) — только если 1.36× мало. Запас около 1.25× сверху.

Ни одна из правок не проверена на полном обучении. Перед заменой рецепта нужен один прогон на 100 эпох, чтобы подтвердить 8.95° и время эпохи. Ожидание для п.3: ~160 s на эпоху вместо 214–222 s; dataloader CGENN-рецепта держал 132 s на эпоху.

## 7. Методика

- Kernel `avgurpetrovich/gatr-flash-clifford-bench` (private, offline RTX, pinned image), версии 2–5.
- Датасеты (private): `avgurpetrovich/ga-research-snapshot-main` (`git archive origin/main`, 9d47e0b) и `avgurpetrovich/flash-clifford-src` (e52775c). Плюс прежние `ga-research-offline-deps` и `ga-research-gatr-deps`.
- Каждая стадия запускалась отдельным процессом, чтобы сорванный CUDA-graph capture не портил следующие замеры.
- Покомпонентные времена (раздел 2) получены повтором каждого перехваченного вызова по отдельности с `triton.testing.do_bench`. Их сумма (89.7 ms) на ~9% больше времени цельного vector field (82.3 ms) из-за изоляции.
- Не проверено:
  - точность обучения с TF32 и bf16;
  - относительная ошибка frozen dense в inference (записана только абсолютная, 1e-5…6e-5, без масштаба выхода);
  - `torch.compile` + TF32 и `torch.compile` + graphsafe вместе.

## 8. В репо: `--gatr_dense_linear` и `--gatr_compile` (ветка `gatr-dense-compile`)

Обновление: позже `--gatr_compile` перешёл на примитивы `gatr_fast_ops`, а GATr стал одним графом. Цифры ниже — первая версия флагов. Текущие (120.5 ms на шаг, с `--backbone_bf16` — 80.3 ms) — в [frameworks.md](frameworks.md), раздел 6.

Из раздела 6 в код вошли п.1 и п.3 без flash gp (у flash-clifford нет лицензии). Оба флага по умолчанию выключены. Веса и state_dict не меняются, так что чекпоинты подходят в обе стороны.

- `--gatr_dense_linear`. `use_dense_linears` меняет класс каждого `EquiLinear` на `DenseEquiLinear`, который делает один `F.linear`.
- `--gatr_compile`. `Module.compile()` на GATr-модулях, при этом:
  - выключен `enable_cached_einsum`;
  - модуль перекомпилируется в `__setstate__`, потому что nn.Module выбрасывает скомпилированный вызов из deepcopy, и EMA-копия иначе шла бы eager;
  - `recompile_limit` поднят до 64: блоки × EMA × train/eval × формы батча не помещались в 8, и dynamo молча уходил в eager.

Замер ветки на RTX PRO 6000 через `train_epoch`, `validate_epoch` и `calculate_evaluation_metrics`, с EMA. Синтетика: 60 батчей по 64 и валидация 6 батчей (последний неполный, 40 картинок). Взята установившаяся эпоха:

| флаги | шаг | eval, 1 сэмпл (360 картинок) | eval, 32 сэмпла |
|---|---|---|---|
| — | 177.2 ms | 1.12 s | 6.54 s |
| `--gatr_dense_linear` | 159.8 ms (−10%) | 1.14 s | 6.72 s |
| `--gatr_compile` | 151.7 ms (−14%) | 0.78 s | 5.79 s |
| оба | **136.3 ms (−23%, 1.30×)** | **0.74 s** | **5.49 s** |

- Разовая компиляция за прогон: ~20 s на первую валидацию, ~12 s на первый 1-сэмпловый eval и ~12–15 s на первый final eval.
- Vector field с флагами против eager при одинаковых весах: относительная разница выхода 7e-8, градиентов ≤2.3e-7 (fp32).
- Прогноз для 100 эпох: ~155 s на train-часть эпохи вместо ~202 s, всего ≈4.6 ч вместо 6.1 ч.
- Полное обучение с флагами ещё не запускалось, 8.95° на них не подтверждены.

