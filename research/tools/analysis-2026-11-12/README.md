# 2026-11-12 の見直しで使った一度きりの分析スクリプト

27B の想起 0/20 の見直し（docs/results/mq-27b-recall-review-2026-11-12.md）と、教師がどの位置で外すかの調査
（docs/results/mq-teacher-positions-2026-11-12.md）で実行したもの。GPU は使わない。
パス（/home/claude/…）は実行した作業環境のもので、そのままでは他の場所で動かない。記録として残す。

| ファイル | したこと |
|---|---|
| emul27.py | 27B のトークナイザの会話の書き出しを、共通の語彙で再現する。VM で記録した 16 行と一致することを確かめる |
| recall_ctx.py | 27B の想起の 80 問を作り直し、質問の前に会話があったかで成否を分ける |
| recall_ctx08.py | 同じことを 0.8B（D1）の 80 問で行う |
| flex_vs_dense.py | 公式のコンパイル済み FlexAttention と密な計算を、128 の倍数でない長さも含めて比べる（CPU） |
| release_scalars_fetch.py | Release の重みから、記憶の頭ごとの係数だけを部分的に読む |
| release_scalars_compare.py | その係数を 0.8B と 27B の各段で比べる |
| eval_rows_documents.py・train_rows_documents.py | 評価の行・学習の行が、どの文書でできているかを再現する |
| teacher_by_context.py | 教師の損失を、会話の前・中・後ろで分けて集計する |

結果のデータ：docs/results/data/mq-27b-recall-review/
